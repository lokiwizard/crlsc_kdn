from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.projection_head import ProjectionHead
from models.timm_encoder import TimmEncoder
from stageB.local_dataset import LocalContrastiveDataset
from stageB.losses import nt_xent_loss
from stageB.transforms import SimCLRTransform


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).parent / "configs" / "train_local_ssl.yaml"
    parser = argparse.ArgumentParser(description="Train Stage B local SimCLR baseline.")
    parser.add_argument("--config", type=str, default=str(default_config), help="YAML config path.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override training subset size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override training device.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config in {path}")
    return config


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    merged = json.loads(json.dumps(config))

    if args.data_root is not None:
        merged["data"]["root"] = args.data_root
    if args.output_dir is not None:
        merged["output"]["dir"] = args.output_dir
    if args.max_samples is not None:
        merged["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        merged["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        merged["train"]["batch_size"] = args.batch_size
    if args.device is not None:
        merged["runtime"]["device"] = args.device

    return merged


def _as_int(value, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} should be an integer, but got boolean: {value}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} should be an integer, but got {value!r}") from exc


def _as_float(value, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} should be a float, but got {value!r}") from exc


def _as_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} should be a boolean, but got {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class SimCLRModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        projection_hidden_dim: int,
        projection_out_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = TimmEncoder(model_name=model_name, pretrained=pretrained)
        self.projector = ProjectionHead(
            in_dim=self.encoder.feature_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(images)
        z = self.projector(h)
        return h, z


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train(config: dict) -> dict:
    seed = _as_int(config["runtime"].get("seed", 42), "runtime.seed")
    device = resolve_device(config["runtime"].get("device", "auto"))
    use_amp = _as_bool(config["runtime"].get("amp", True), "runtime.amp")

    data_root = config["data"]["root"]
    max_samples = _as_int(config["data"].get("max_samples"), "data.max_samples")
    subset_seed = _as_int(config["data"].get("subset_seed", 42), "data.subset_seed")
    image_size = _as_int(config["data"].get("image_size", 224), "data.image_size")

    model_name = config["model"].get("name", "resnet18")
    pretrained = _as_bool(config["model"].get("pretrained", False), "model.pretrained")
    projection_hidden_dim = _as_int(config["model"].get("projection_hidden_dim", 2048), "model.projection_hidden_dim")
    projection_out_dim = _as_int(config["model"].get("projection_out_dim", 128), "model.projection_out_dim")

    epochs = _as_int(config["train"].get("epochs", 100), "train.epochs")
    batch_size = _as_int(config["train"].get("batch_size", 128), "train.batch_size")
    num_workers = _as_int(config["train"].get("num_workers", 0), "train.num_workers")
    temperature = _as_float(config["train"].get("temperature", 0.2), "train.temperature")
    lr = _as_float(config["optimizer"].get("lr", 1e-3), "optimizer.lr")
    weight_decay = _as_float(config["optimizer"].get("weight_decay", 1e-4), "optimizer.weight_decay")
    checkpoint_interval = _as_int(config["output"].get("checkpoint_interval", 10), "output.checkpoint_interval")
    output_dir = Path(config["output"]["dir"])

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    transform = SimCLRTransform(image_size=image_size)
    dataset = LocalContrastiveDataset(
        root=data_root,
        transform=transform,
        max_samples=max_samples,
        subset_seed=subset_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= batch_size,
    )

    if len(loader) == 0:
        raise RuntimeError(
            f"Training loader is empty. dataset_size={len(dataset)}, batch_size={batch_size}. "
            "Reduce batch size or increase max_samples."
        )

    model = SimCLRModel(
        model_name=model_name,
        pretrained=pretrained,
        projection_hidden_dim=projection_hidden_dim,
        projection_out_dim=projection_out_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    history = []
    best_loss = float("inf")

    run_config = {
        "data": {
            "root": str(Path(data_root).resolve()),
            "max_samples": max_samples,
            "subset_seed": subset_seed,
            "image_size": image_size,
            "num_samples": len(dataset),
        },
        "model": {
            "name": model_name,
            "pretrained": pretrained,
            "encoder_dim": model.encoder.feature_dim,
            "projection_hidden_dim": projection_hidden_dim,
            "projection_out_dim": projection_out_dim,
        },
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "temperature": temperature,
        },
        "optimizer": {
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "runtime": {
            "device": str(device),
            "amp": use_amp and device.type == "cuda",
            "seed": seed,
        },
    }
    save_json(output_dir / "run_config.json", run_config)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        sample_count = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss, acc = nt_xent_loss(z1, z2, temperature=temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_now = view1.shape[0]
            epoch_loss += loss.item() * batch_size_now
            epoch_acc += acc * batch_size_now
            sample_count += batch_size_now
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        scheduler.step()

        avg_loss = epoch_loss / sample_count
        avg_acc = epoch_acc / sample_count
        lr_now = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "loss": avg_loss,
            "contrastive_accuracy": avg_acc,
            "lr": lr_now,
        }
        history.append(record)
        save_json(output_dir / "history.json", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "projector_state_dict": model.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "config": run_config,
        }

        torch.save(checkpoint, output_dir / "checkpoints" / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, output_dir / "checkpoints" / "best.pt")
        if checkpoint_interval is not None and checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            torch.save(checkpoint, output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

    encoder_state = {
        "encoder_state_dict": model.encoder.state_dict(),
        "feature_dim": model.encoder.feature_dim,
        "model_name": model_name,
        "pretrained": pretrained,
    }
    torch.save(encoder_state, output_dir / "encoder_final.pt")

    summary = {
        "output_dir": str(output_dir.resolve()),
        "num_samples": len(dataset),
        "epochs": epochs,
        "best_loss": best_loss,
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["contrastive_accuracy"],
        "device": str(device),
    }
    save_json(output_dir / "train_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    merged_config = merge_cli_overrides(config, args)
    summary = train(merged_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
