"""Local text embeddings (fastembed, ONNX). The model downloads once and is cached under ``data/models``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384-d, handles Russian notes
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def _model(cache_dir: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir)


def embed(text: str, cache_dir: Path) -> list[float]:
    """Embed one text into a ``VECTOR_SIZE`` float vector."""
    return [float(v) for v in next(iter(_model(str(cache_dir)).embed([text])))]
