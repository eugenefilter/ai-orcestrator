"""Knowledge-base service shared by the CLI, the web API and the MCP server."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
from pydantic import ValidationError
from qdrant_client import QdrantClient

from .analysis.report import ANALYZER_VERSION, write_report
from .config import Settings
from .store import qdrant_store as qs
from .store.embed import embed
from .store.schema import Entry, PatchRecipe, embedding_text, normalize_tags

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3"}
NAMESPACE = uuid.UUID("5b0f0f3e-6c3a-4a53-9a0e-0d5a7e6f1b11")
MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 60.0


class KBError(ValueError):
    """A user-facing problem (bad path, bad format, unknown id)."""


def resolve_allowed(value: str | Path, settings: Settings) -> Path:
    """Resolve symlinks and ``..``, then require an existing audio file under ``ALLOWED_INGEST_DIRS`` or ``DATA_DIR`` (renders)."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise KBError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise KBError("Supported formats: WAV, AIFF, FLAC, MP3.")
    if not any(path.is_relative_to(root) for root in (*settings.allowed_ingest_dirs, settings.data_dir)):
        raise KBError("Path is outside ALLOWED_INGEST_DIRS (see .env).")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class KnowledgeBase:
    """Sounds, notes and device maps: files under ``data/`` plus vectors and payload in Qdrant."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.client = client or qs.connect(settings.qdrant_url)
        try:
            qs.ensure_collection(self.client)
        except qs.ModelMismatch:
            raise
        except Exception as exc:
            raise KBError(f"Qdrant is not reachable at {settings.qdrant_url}. Run `docker compose up -d`. ({exc})") from exc
        self.audio_dir = settings.data_dir / "audio"
        self.analysis_dir = settings.data_dir / "analysis"

    # -- writing -------------------------------------------------------------------------------

    def _save(self, point_id: str, entry: Entry) -> dict[str, Any]:
        entry.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        qs.upsert(self.client, point_id, embed(embedding_text(entry), self.settings.data_dir / "models"), entry.model_dump(mode="json"))
        return {"id": point_id, **entry.model_dump(mode="json")}

    def ingest_audio(self, path: Path, title: str | None = None, tags: list[str] | None = None, notes: str = "",
                     recipes: list[PatchRecipe] | None = None, allow_long: bool = False, source_path: str | None = None) -> dict[str, Any]:
        """Analyze ``path``, copy it under ``data/audio`` and store it. Same file content updates the same entry."""
        if path.stat().st_size > MAX_BYTES:
            raise KBError("File is larger than 100 MB.")
        try:
            info = sf.info(path)
        except Exception as exc:
            raise KBError(f"Could not read audio file '{path.name}': {exc}") from exc
        if info.duration > MAX_SECONDS and not allow_long:
            raise KBError(f"Clip is {info.duration:.0f} s; limit is {MAX_SECONDS:.0f} s (use allow_long).")
        digest = sha256(path)
        point_id = str(uuid.uuid5(NAMESPACE, digest))
        stored = self.audio_dir / f"{digest}{path.suffix.lower()}"
        stored.parent.mkdir(parents=True, exist_ok=True)
        if not stored.exists():
            shutil.copyfile(path, stored)
        report = write_report(stored, self.analysis_dir / digest)
        previous = qs.get(self.client, point_id)
        entry = Entry.model_validate(previous) if previous else Entry(kind="sound", title=title or path.stem)
        if title:
            entry.title = title
        entry.tags = normalize_tags(entry.tags + (tags or []))
        entry.notes = notes or entry.notes
        entry.audio_path = str(stored)
        entry.source_path = source_path or str(path)
        entry.analysis = report
        entry.patch_recipes += recipes or []
        return self._save(point_id, entry)

    def add_text(self, kind: str, title: str, text: str, tags: list[str] | None = None,
                 recipes: list[PatchRecipe] | None = None) -> dict[str, Any]:
        """Store a ``note`` or ``device_map`` entry (no audio)."""
        if kind not in ("note", "device_map"):
            raise KBError("kind must be 'note' or 'device_map'.")
        return self._save(str(uuid.uuid4()), Entry(kind=kind, title=title, notes=text, tags=tags or [], patch_recipes=recipes or []))

    def update(self, point_id: str, title: str | None = None, tags: list[str] | None = None, notes: str | None = None,
               recipes: list[PatchRecipe] | None = None) -> dict[str, Any]:
        """Replace the given fields of an entry (``None`` keeps the old value) and re-embed it."""
        entry = self._load(point_id)
        if title is not None:
            entry.title = title
        if tags is not None:
            entry.tags = normalize_tags(tags)
        if notes is not None:
            entry.notes = notes
        if recipes is not None:
            entry.patch_recipes = recipes
        return self._save(point_id, entry)

    def add_recipe(self, point_id: str, recipe: PatchRecipe) -> dict[str, Any]:
        """Append a recipe, or replace the one with the same name."""
        entry = self._load(point_id)
        entry.patch_recipes = [r for r in entry.patch_recipes if r.name != recipe.name] + [recipe]
        return self._save(point_id, entry)

    def reanalyze(self, point_id: str) -> dict[str, Any]:
        """Re-run the current analysis on the stored audio and refresh the entry (tags, notes and recipes are kept)."""
        entry = self._load(point_id)
        if not entry.audio_path or not Path(entry.audio_path).is_file():
            raise KBError("This entry has no stored audio to analyze.")
        entry.analysis = write_report(Path(entry.audio_path), self.analysis_dir / Path(entry.audio_path).stem)
        return self._save(point_id, entry)

    def reanalyze_outdated(self, force: bool = False) -> list[dict[str, Any]]:
        """Re-analyze every sound whose report predates ``ANALYZER_VERSION`` (all sounds with ``force``)."""
        done = []
        for point_id, payload in qs.scroll(self.client, qs.build_filter(kind="sound")):
            if force or (payload.get("analysis") or {}).get("analyzer", 1) < ANALYZER_VERSION:
                done.append({"id": point_id, "title": payload.get("title")} | {"summary": self.reanalyze(point_id)["analysis"]["summary"]})
        return done

    def delete(self, point_id: str) -> None:
        entry = self._load(point_id)
        if entry.audio_path:
            digest = Path(entry.audio_path).stem
            Path(entry.audio_path).unlink(missing_ok=True)
            shutil.rmtree(self.analysis_dir / digest, ignore_errors=True)
        qs.delete(self.client, point_id)

    # -- reading -------------------------------------------------------------------------------

    def _load(self, point_id: str) -> Entry:
        try:
            point_id = str(uuid.UUID(point_id))
        except ValueError as exc:
            raise KBError(f"No entry with id {point_id}.") from exc
        payload = qs.get(self.client, point_id)
        if payload is None:
            raise KBError(f"No entry with id {point_id}.")
        return Entry.model_validate(payload)

    def get(self, point_id: str) -> dict[str, Any]:
        return {"id": point_id, **self._load(point_id).model_dump(mode="json")}

    def search(self, query: str = "", kind: str | None = None, tags: list[str] | None = None, any_tag: bool = False,
               f0_range: tuple[float, float] | None = None, duration_range: tuple[float, float] | None = None,
               limit: int = 20) -> list[dict[str, Any]]:
        """Semantic search when ``query`` is given, otherwise a filtered listing (newest first)."""
        flt = qs.build_filter(kind, normalize_tags(tags or []), any_tag, f0_range, duration_range)
        if query.strip():
            rows = [(i, s, p) for i, s, p in qs.search(self.client, embed(query, self.settings.data_dir / "models"), flt, limit)]
        else:
            rows = sorted(((i, None, p) for i, p in qs.scroll(self.client, flt)), key=lambda r: r[2].get("updated_at", ""), reverse=True)[:limit]
        return [self._card(i, s, p) for i, s, p in rows]

    @staticmethod
    def _card(point_id: str, score: float | None, payload: dict[str, Any]) -> dict[str, Any]:
        analysis = payload.get("analysis") or {}
        return {"id": point_id, "score": None if score is None else round(score, 3), "kind": payload.get("kind"),
                "title": payload.get("title"), "tags": payload.get("tags", []), "notes": payload.get("notes", ""),
                "summary": analysis.get("summary", []), "recipes": [r["name"] for r in payload.get("patch_recipes", [])],
                "has_audio": bool(payload.get("audio_path")),
                "outdated": bool(analysis) and analysis.get("analyzer", 1) < ANALYZER_VERSION, "updated_at": payload.get("updated_at")}

    def tags(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for _, payload in qs.scroll(self.client, None):
            for tag in payload.get("tags", []):
                counts[tag] = counts.get(tag, 0) + 1
        return [{"tag": t, "count": n} for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def report_png(self, point_id: str) -> Path | None:
        entry = self._load(point_id)
        return self.analysis_dir / Path(entry.audio_path).stem / "report.png" if entry.audio_path else None


def validate_recipe(data: dict[str, Any]) -> PatchRecipe:
    """Validate recipe data; raises ``KBError`` listing each problem as ``path: message``."""
    try:
        return PatchRecipe.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise KBError(f"Recipe is invalid: {problems}") from exc


def parse_recipe(text: str) -> PatchRecipe:
    """Parse and validate recipe JSON."""
    try:
        return validate_recipe(json.loads(text))
    except json.JSONDecodeError as exc:
        raise KBError(f"Recipe is not valid JSON: {exc}") from exc
