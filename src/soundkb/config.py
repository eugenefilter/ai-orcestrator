"""Configuration loading for SoundKB."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for local SoundKB services."""

    qdrant_url: str
    data_dir: Path
    ui_port: int
    allowed_ingest_dirs: tuple[Path, ...]


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from ``.env`` and environment variables.

    Relative ``DATA_DIR`` values are resolved from the current working directory.
    """
    load_dotenv(dotenv_path=env_file, override=False)
    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
    raw_dirs = os.getenv("ALLOWED_INGEST_DIRS", "~/Documents/Ableton:~/Music")
    allowed_dirs = tuple(
        Path(item).expanduser().resolve() for item in raw_dirs.split(":") if item.strip()
    )
    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        data_dir=data_dir,
        ui_port=int(os.getenv("UI_PORT", "8766")),
        allowed_ingest_dirs=allowed_dirs,
    )
