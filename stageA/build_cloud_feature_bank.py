from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models.clip_encoder import CLIPImageEncoder
from models.feature_bank import save_feature_bank
from models.retriever import FeatureRetriever
from stageA.image_dataset import ImagePathDataset


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).parent / "configs" / "build_feature_bank.yaml"
    parser = argparse.ArgumentParser(description="Build the Stage A cloud CLIP feature bank.")
    parser.add_argument("--config", type=str, default=str(default_config), help="YAML config path.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--model-name", type=str, default=None, help="Override model id or local model directory.")
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Override checkpoint root.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override max_samples for smoke tests.")
    parser.add_argument("--local-files-only", action="store_true", help="Force local-only model loading.")
    parser.add_argument("--no-faiss", action="store_true", help="Disable FAISS even if enabled in YAML.")
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
    if args.model_name is not None:
        merged["model"]["name"] = args.model_name
    if args.checkpoint_root is not None:
        merged["model"]["checkpoint_root"] = args.checkpoint_root
    if args.device is not None:
        merged["runtime"]["device"] = args.device
    if args.batch_size is not None:
        merged["runtime"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        merged["runtime"]["num_workers"] = args.num_workers
    if args.max_samples is not None:
        merged["data"]["max_samples"] = args.max_samples
    if args.local_files_only:
        merged["model"]["local_files_only"] = True
    if args.no_faiss:
        merged["retriever"]["use_faiss"] = False

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


def _progress(iterable, total: int):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="Encoding")
    except Exception:
        return iterable


def build_feature_bank(config: dict) -> dict[str, str | int | None]:
    data_root = config["data"]["root"]
    output_dir = config["output"]["dir"]
    model_name = config["model"]["name"]
    checkpoint_root = config["model"]["checkpoint_root"]
    local_files_only = _as_bool(config["model"].get("local_files_only", False), "model.local_files_only")
    device = config["runtime"].get("device", "auto")
    batch_size = _as_int(config["runtime"].get("batch_size", 64), "runtime.batch_size")
    num_workers = _as_int(config["runtime"].get("num_workers", 0), "runtime.num_workers")
    max_samples = _as_int(config["data"].get("max_samples"), "data.max_samples")
    use_faiss = _as_bool(config["retriever"].get("use_faiss", True), "retriever.use_faiss")

    encoder = CLIPImageEncoder.load(
        model_name=model_name,
        checkpoint_root=checkpoint_root,
        device=device,
        local_files_only=local_files_only,
    )

    dataset = ImagePathDataset(data_root, transform=encoder.preprocess_image)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=encoder.device.type == "cuda",
    )

    all_features = []
    metadata: list[dict] = []

    for batch in _progress(loader, total=len(loader)):
        images = batch["image"]
        features = encoder.encode(images).cpu().numpy().astype(np.float32)
        all_features.append(features)

        batch_size = features.shape[0]
        for offset in range(batch_size):
            metadata.append(
                {
                    "feature_index": len(metadata),
                    "dataset_index": int(batch["index"][offset]),
                    "rel_path": batch["rel_path"][offset],
                    "abs_path": batch["abs_path"][offset],
                    "class_name": batch["class_name"][offset],
                    "label": int(batch["label"][offset]),
                }
            )

    features_np = np.concatenate(all_features, axis=0)
    config = {
        "stage": "A",
        "data_root": str(Path(data_root).resolve()),
        "num_samples": len(dataset),
        "feature_dim": int(features_np.shape[1]),
        "model_name": model_name,
        "checkpoint_root": str(Path(checkpoint_root).resolve()),
        "backend": "transformers",
        "device": str(encoder.device),
    }

    saved = save_feature_bank(output_dir, features_np, metadata, config)
    retriever = FeatureRetriever.build(features_np, prefer_faiss=use_faiss)
    index_path = retriever.save(output_dir)

    summary = {
        **saved,
        "index": index_path,
        "retriever_backend": retriever.backend,
        "num_samples": len(dataset),
        "feature_dim": int(features_np.shape[1]),
    }

    summary_path = Path(output_dir) / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    merged_config = merge_cli_overrides(config, args)
    summary = build_feature_bank(merged_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
