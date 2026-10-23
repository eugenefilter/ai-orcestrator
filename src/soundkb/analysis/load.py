"""Safe audio loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load audio as mono and original channels, both float32; preserves sample rate."""
    try:
        channels, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not read audio file '{path}': {exc}") from exc
    if channels.size == 0:
        raise ValueError(f"Audio file '{path}' contains no samples.")
    mono = channels.mean(axis=1, dtype=np.float32)
    return mono, channels, int(sr)
