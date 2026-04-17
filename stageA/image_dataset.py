from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._scan(self.root)
        self.class_to_idx = self._build_class_index(self.samples)

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
                    "abs_path": str(path.resolve()),
                    "rel_path": rel_path.as_posix(),
                    "class_name": class_name,
                }
            )

        if not samples:
            raise RuntimeError(f"No images found under {root}")

        return samples

    @staticmethod
    def _build_class_index(samples: list[dict]) -> dict[str, int]:
        class_names = sorted({sample["class_name"] for sample in samples if sample["class_name"]})
        return {name: idx for idx, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["abs_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        class_name = sample["class_name"]
        label = self.class_to_idx.get(class_name, -1)
        return {
            "image": image,
            "index": index,
            "abs_path": sample["abs_path"],
            "rel_path": sample["rel_path"],
            "class_name": class_name,
            "label": label,
        }
