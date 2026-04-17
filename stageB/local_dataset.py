from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from stageA.image_dataset import IMAGE_EXTENSIONS


class LocalContrastiveDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform,
        max_samples: int | None = None,
        subset_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._scan(self.root)

        if max_samples is not None and max_samples < len(self.samples):
            rng = random.Random(subset_seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            selected = sorted(indices[:max_samples])
            self.samples = [self.samples[idx] for idx in selected]

    @staticmethod
    def _scan(root: Path) -> list[dict]:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        samples = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel_path = path.relative_to(root)
            class_name = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
            samples.append(
                {
                    "index": len(samples),
                    "abs_path": str(path.resolve()),
                    "rel_path": rel_path.as_posix(),
                    "class_name": class_name,
                }
            )

        if not samples:
            raise RuntimeError(f"No images found under {root}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample["abs_path"]).convert("RGB")
        view1, view2 = self.transform(image)
        return {
            "view1": view1,
            "view2": view2,
            "index": sample["index"],
            "abs_path": sample["abs_path"],
            "rel_path": sample["rel_path"],
            "class_name": sample["class_name"],
        }
