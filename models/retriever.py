from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.feature_bank import ensure_normalized


class FeatureRetriever:
    def __init__(self, features: np.ndarray, backend: str, index=None):
        self.features = ensure_normalized(features)
        self.backend = backend
        self.index = index

    @classmethod
    def build(cls, features: np.ndarray, prefer_faiss: bool = True) -> "FeatureRetriever":
        features = ensure_normalized(features)

        if prefer_faiss:
            try:
                import faiss

                index = faiss.IndexFlatIP(features.shape[1])
                index.add(np.ascontiguousarray(features))
                return cls(features=features, backend="faiss", index=index)
            except Exception:
                pass

        return cls(features=features, backend="numpy")

    def save(self, output_dir: str | Path) -> str | None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        if self.backend == "faiss":
            import faiss

            index_path = root / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            return str(index_path)

        fallback_path = root / "retriever_meta.json"
        fallback_path.write_text(json.dumps({"backend": "numpy"}, indent=2), encoding="utf-8")
        return None

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = ensure_normalized(queries)

        if self.backend == "faiss":
            scores, indices = self.index.search(np.ascontiguousarray(queries), k)
            return scores, indices

        scores = queries @ self.features.T
        top_indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores, top_indices

    @classmethod
    def load(cls, output_dir: str | Path, prefer_faiss: bool = True) -> "FeatureRetriever":
        root = Path(output_dir)
        features = np.load(root / "cloud_features.npy").astype(np.float32)

        if prefer_faiss and (root / "faiss.index").exists():
            try:
                import faiss

                index = faiss.read_index(str(root / "faiss.index"))
                return cls(features=features, backend="faiss", index=index)
            except Exception:
                pass

        return cls(features=features, backend="numpy")
