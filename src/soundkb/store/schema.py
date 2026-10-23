"""Pydantic models for knowledge-base entries and Ableton patch recipes."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def normalize_tag(tag: str) -> str:
    """Lower-case, trim and replace whitespace with ``-``."""
    return re.sub(r"\s+", "-", tag.strip().lower())


def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize, drop empties and de-duplicate while keeping order."""
    return list(dict.fromkeys(t for t in map(normalize_tag, tags) if t))


class PatchTarget(BaseModel):
    """The Ableton device a recipe is built on."""

    live_version: str = "11.3"
    device: str = Field(min_length=1)
    browser_uri: str | None = None


class PatchParam(BaseModel):
    """One device parameter exactly as the Ableton MCP reports and accepts it."""

    name: str = Field(min_length=1)
    index: int | None = None
    value: float
    value_display: str | None = None
    note: str | None = None


class ChainDevice(BaseModel):
    """A further top-level device on the same track."""

    device: str = Field(min_length=1)
    browser_uri: str | None = None
    params: list[PatchParam] = []


class MidiSpec(BaseModel):
    note: int = Field(ge=0, le=127)
    velocity: int = Field(default=100, ge=1, le=127)
    length_beats: float = Field(default=4, gt=0)


class Verification(BaseModel):
    distance: float
    iterations: int
    date: str
    render_path: str | None = None


class PatchRecipe(BaseModel):
    """How to recreate a sound in Ableton: synth, parameters, chain and test note."""

    name: str = Field(min_length=1)
    target: PatchTarget
    params: list[PatchParam] = []
    chain: list[ChainDevice] = []
    midi: MidiSpec | None = None
    verified: Verification | None = None
    confidence: Literal["guess", "close", "verified"] = "guess"


class Entry(BaseModel):
    """A knowledge-base entry; this is exactly the Qdrant payload."""

    kind: Literal["sound", "note", "device_map"]
    title: str = Field(min_length=1)
    tags: list[str] = []
    notes: str = ""
    audio_path: str | None = None
    source_path: str | None = None
    analysis: dict[str, Any] | None = None
    patch_recipes: list[PatchRecipe] = []
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str]) -> list[str]:
        return normalize_tags(value)


def embedding_text(entry: Entry) -> str:
    """Text that represents an entry in the vector index."""
    parts = [entry.title, " ".join(entry.tags), entry.notes]
    if entry.analysis:
        parts.append(" ".join(entry.analysis.get("summary", [])))
    parts += [recipe.name + " " + recipe.target.device for recipe in entry.patch_recipes]
    return "\n".join(p for p in parts if p)
