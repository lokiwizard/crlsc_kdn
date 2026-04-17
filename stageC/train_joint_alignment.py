from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.alignment_head import AlignmentHead
from models.feature_bank import load_feature_bank
from models.projection_head import ProjectionHead
from models.retriever import FeatureRetriever
from models.timm_encoder import TimmEncoder
from stageB.local_dataset import LocalContrastiveDataset
from stageB.losses import nt_xent_loss
from stageB.transforms import SimCLRTransform


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).parent / "configs" / "train_joint_alignment.yaml"
    parser = argparse.ArgumentParser(description="Train Stage C local model with cloud semantic alignment.")
    parser.add_argument("--config", type=str, default=str(default_config), help="YAML config path.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir.")
    parser.add_argument("--feature-bank-dir", type=str, default=None, help="Override Stage A feature bank dir.")
    parser.add_argument("--pretrain-checkpoint", type=str, default=None, help="Override Stage B checkpoint path.")
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
    if args.feature_bank_dir is not None:
        merged["cloud"]["feature_bank_dir"] = args.feature_bank_dir
    if args.pretrain_checkpoint is not None:
        merged["init"]["stageb_checkpoint"] = args.pretrain_checkpoint
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


def resolve_existing_path(path_value: str | Path, field_name: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    fallback = path.parent / path.parent.name / path.name
    if fallback.exists():
        return fallback

    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part.lower() in {"stagea", "stageb", "stagec"}:
            duplicated = Path(*parts[: idx + 1], part, *parts[idx + 1 :])
            if duplicated.exists():
                return duplicated

    raise FileNotFoundError(f"{field_name} does not exist: {path}")


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class JointAlignmentModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        projection_hidden_dim: int,
        projection_out_dim: int,
        alignment_out_dim: int,
        alignment_hidden_dim: int | None,
    ) -> None:
        super().__init__()
        self.encoder = TimmEncoder(model_name=model_name, pretrained=pretrained)
        self.projector = ProjectionHead(
            in_dim=self.encoder.feature_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )
        self.aligner = AlignmentHead(
            in_dim=self.encoder.feature_dim,
            out_dim=alignment_out_dim,
            hidden_dim=alignment_hidden_dim,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(images)
        z = self.projector(h)
        q = self.aligner(h)
        return h, z, q


def load_stageb_checkpoint(
    model: JointAlignmentModel,
    checkpoint_path: str | Path,
    expected_model_name: str,
    expected_pretrained: bool,
    expected_projection_hidden_dim: int,
    expected_projection_out_dim: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model_cfg = checkpoint.get("config", {}).get("model", {})

    if checkpoint_model_cfg:
        ckpt_model_name = checkpoint_model_cfg.get("name")
        ckpt_pretrained = checkpoint_model_cfg.get("pretrained")
        ckpt_proj_hidden = checkpoint_model_cfg.get("projection_hidden_dim")
        ckpt_proj_out = checkpoint_model_cfg.get("projection_out_dim")

        mismatches = []
        if ckpt_model_name != expected_model_name:
            mismatches.append(f"model.name checkpoint={ckpt_model_name} current={expected_model_name}")
        if ckpt_pretrained != expected_pretrained:
            mismatches.append(f"model.pretrained checkpoint={ckpt_pretrained} current={expected_pretrained}")
        if ckpt_proj_hidden != expected_projection_hidden_dim:
            mismatches.append(
                f"model.projection_hidden_dim checkpoint={ckpt_proj_hidden} current={expected_projection_hidden_dim}"
            )
        if ckpt_proj_out != expected_projection_out_dim:
            mismatches.append(f"model.projection_out_dim checkpoint={ckpt_proj_out} current={expected_projection_out_dim}")

        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(f"Stage B checkpoint is incompatible with current Stage C config: {details}")

    if "encoder_state_dict" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    if "projector_state_dict" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector_state_dict"], strict=True)


def compute_semantic_prototype(
    queries: torch.Tensor,
    retriever: FeatureRetriever,
    top_k: int,
    retrieval_temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_np = queries.detach().cpu().numpy().astype(np.float32)
    scores_np, indices_np = retriever.search(query_np, k=top_k)

    neighbor_np = retriever.features[indices_np]
    scores = torch.from_numpy(scores_np).to(device=device, dtype=torch.float32)
    neighbors = torch.from_numpy(neighbor_np).to(device=device, dtype=torch.float32)

    weights = torch.softmax(scores / retrieval_temperature, dim=1)
    prototypes = torch.sum(weights.unsqueeze(-1) * neighbors, dim=1)
    prototypes = F.normalize(prototypes, dim=-1)
    return prototypes, weights, torch.from_numpy(indices_np).to(device=device)


def retrieval_top1_accuracy(
    top_indices: torch.Tensor,
    batch_class_names,
    cloud_metadata: list[dict],
) -> float:
    if top_indices.numel() == 0:
        return 0.0

    correct = 0
    total = 0
    top1_indices = top_indices[:, 0].detach().cpu().tolist()
    for local_class, cloud_idx in zip(batch_class_names, top1_indices):
        if not local_class:
            continue
        cloud_class = cloud_metadata[int(cloud_idx)].get("class_name", "")
        if not cloud_class:
            continue
        total += 1
        if cloud_class == local_class:
            correct += 1

    if total == 0:
        return 0.0
    return correct / total


def cosine_alignment_loss(queries: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return (1.0 - F.cosine_similarity(queries, prototypes, dim=-1)).mean()


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
    alignment_hidden_dim = _as_int(config["model"].get("alignment_hidden_dim"), "model.alignment_hidden_dim")

    epochs = _as_int(config["train"].get("epochs", 100), "train.epochs")
    batch_size = _as_int(config["train"].get("batch_size", 128), "train.batch_size")
    num_workers = _as_int(config["train"].get("num_workers", 0), "train.num_workers")
    temperature = _as_float(config["train"].get("temperature", 0.2), "train.temperature")
    lambda_align = _as_float(config["train"].get("lambda_align", 0.5), "train.lambda_align")
    align_both_views = _as_bool(config["train"].get("align_both_views", True), "train.align_both_views")
    lr = _as_float(config["optimizer"].get("lr", 1e-3), "optimizer.lr")
    weight_decay = _as_float(config["optimizer"].get("weight_decay", 1e-4), "optimizer.weight_decay")
    checkpoint_interval = _as_int(config["output"].get("checkpoint_interval", 10), "output.checkpoint_interval")
    output_dir = Path(config["output"]["dir"])

    feature_bank_dir = resolve_existing_path(config["cloud"]["feature_bank_dir"], "cloud.feature_bank_dir")
    top_k = _as_int(config["cloud"].get("top_k", 10), "cloud.top_k")
    retrieval_temperature = _as_float(config["cloud"].get("retrieval_temperature", 0.07), "cloud.retrieval_temperature")
    prefer_faiss = _as_bool(config["cloud"].get("use_faiss", True), "cloud.use_faiss")

    stageb_checkpoint = config.get("init", {}).get("stageb_checkpoint")
    load_stageb = _as_bool(config.get("init", {}).get("load_stageb_checkpoint", True), "init.load_stageb_checkpoint")
    if load_stageb and stageb_checkpoint:
        stageb_checkpoint = resolve_existing_path(stageb_checkpoint, "init.stageb_checkpoint")

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    cloud_features, cloud_metadata, cloud_config = load_feature_bank(feature_bank_dir)
    retriever = FeatureRetriever.load(feature_bank_dir, prefer_faiss=prefer_faiss)
    cloud_dim = int(cloud_features.shape[1])

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

    model = JointAlignmentModel(
        model_name=model_name,
        pretrained=pretrained,
        projection_hidden_dim=projection_hidden_dim,
        projection_out_dim=projection_out_dim,
        alignment_out_dim=cloud_dim,
        alignment_hidden_dim=alignment_hidden_dim,
    ).to(device)

    if load_stageb and stageb_checkpoint:
        load_stageb_checkpoint(
            model,
            stageb_checkpoint,
            expected_model_name=model_name,
            expected_pretrained=pretrained,
            expected_projection_hidden_dim=projection_hidden_dim,
            expected_projection_out_dim=projection_out_dim,
        )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    history = []
    best_total_loss = float("inf")

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
            "alignment_hidden_dim": alignment_hidden_dim,
            "alignment_out_dim": cloud_dim,
        },
        "cloud": {
            "feature_bank_dir": str(feature_bank_dir.resolve()),
            "num_cloud_samples": len(cloud_metadata),
            "cloud_feature_dim": cloud_dim,
            "retriever_backend": retriever.backend,
            "top_k": top_k,
            "retrieval_temperature": retrieval_temperature,
            "stagea_model_name": cloud_config.get("model_name"),
        },
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "temperature": temperature,
            "lambda_align": lambda_align,
            "align_both_views": align_both_views,
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
        "init": {
            "load_stageb_checkpoint": load_stageb,
            "stageb_checkpoint": str(Path(stageb_checkpoint).resolve()) if stageb_checkpoint else None,
        },
    }
    save_json(output_dir / "run_config.json", run_config)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_ssl_loss = 0.0
        epoch_align_loss = 0.0
        epoch_ssl_acc = 0.0
        epoch_neighbor_score = 0.0
        epoch_retrieval_acc = 0.0
        sample_count = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)
            batch_class_names = batch["class_name"]

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                _, z1, q1 = model(view1)
                _, z2, q2 = model(view2)
                ssl_loss, ssl_acc = nt_xent_loss(z1, z2, temperature=temperature)

                prototypes1, weights1, top_indices1 = compute_semantic_prototype(
                    F.normalize(q1, dim=-1),
                    retriever=retriever,
                    top_k=top_k,
                    retrieval_temperature=retrieval_temperature,
                    device=device,
                )
                retrieval_acc = retrieval_top1_accuracy(top_indices1, batch_class_names, cloud_metadata)
                align_loss = cosine_alignment_loss(q1, prototypes1)

                if align_both_views:
                    prototypes2, weights2, _ = compute_semantic_prototype(
                        F.normalize(q2, dim=-1),
                        retriever=retriever,
                        top_k=top_k,
                        retrieval_temperature=retrieval_temperature,
                        device=device,
                    )
                    align_loss = 0.5 * (align_loss + cosine_alignment_loss(q2, prototypes2))
                    mean_neighbor_score = 0.5 * (weights1[:, 0].mean() + weights2[:, 0].mean())
                else:
                    mean_neighbor_score = weights1[:, 0].mean()

                total_loss = ssl_loss + lambda_align * align_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_now = view1.shape[0]
            epoch_total_loss += total_loss.item() * batch_size_now
            epoch_ssl_loss += ssl_loss.item() * batch_size_now
            epoch_align_loss += align_loss.item() * batch_size_now
            epoch_ssl_acc += ssl_acc * batch_size_now
            epoch_neighbor_score += float(mean_neighbor_score.item()) * batch_size_now
            epoch_retrieval_acc += retrieval_acc * batch_size_now
            sample_count += batch_size_now
            progress.set_postfix(
                total=f"{total_loss.item():.4f}",
                ssl=f"{ssl_loss.item():.4f}",
                align=f"{align_loss.item():.4f}",
                acc=f"{ssl_acc:.4f}",
                r1=f"{retrieval_acc:.4f}",
            )

        scheduler.step()

        avg_total_loss = epoch_total_loss / sample_count
        avg_ssl_loss = epoch_ssl_loss / sample_count
        avg_align_loss = epoch_align_loss / sample_count
        avg_ssl_acc = epoch_ssl_acc / sample_count
        avg_neighbor_score = epoch_neighbor_score / sample_count
        avg_retrieval_acc = epoch_retrieval_acc / sample_count
        lr_now = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "total_loss": avg_total_loss,
            "ssl_loss": avg_ssl_loss,
            "align_loss": avg_align_loss,
            "acc": avg_ssl_acc,
            "contrastive_accuracy": avg_ssl_acc,
            "retrieval_top1_acc": avg_retrieval_acc,
            "mean_top1_neighbor_weight": avg_neighbor_score,
            "lr": lr_now,
        }
        history.append(record)
        save_json(output_dir / "history.json", history)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "projector_state_dict": model.projector.state_dict(),
            "aligner_state_dict": model.aligner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "config": run_config,
        }

        torch.save(checkpoint, output_dir / "checkpoints" / "last.pt")
        if avg_total_loss < best_total_loss:
            best_total_loss = avg_total_loss
            torch.save(checkpoint, output_dir / "checkpoints" / "best.pt")
        if checkpoint_interval is not None and checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            torch.save(checkpoint, output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

    encoder_state = {
        "encoder_state_dict": model.encoder.state_dict(),
        "aligner_state_dict": model.aligner.state_dict(),
        "feature_dim": model.encoder.feature_dim,
        "cloud_dim": cloud_dim,
        "model_name": model_name,
        "pretrained": pretrained,
    }
    torch.save(encoder_state, output_dir / "encoder_aligned_final.pt")

    summary = {
        "output_dir": str(output_dir.resolve()),
        "num_samples": len(dataset),
        "epochs": epochs,
        "best_total_loss": best_total_loss,
        "final_total_loss": history[-1]["total_loss"],
        "final_ssl_loss": history[-1]["ssl_loss"],
        "final_align_loss": history[-1]["align_loss"],
        "acc": history[-1]["acc"],
        "final_accuracy": history[-1]["contrastive_accuracy"],
        "final_retrieval_top1_acc": history[-1]["retrieval_top1_acc"],
        "retriever_backend": retriever.backend,
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
