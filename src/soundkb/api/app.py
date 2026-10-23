"""Local web interface for SoundKB (binds to 127.0.0.1 only)."""

from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from soundkb.config import load_settings
from soundkb.kb import AUDIO_SUFFIXES, MAX_BYTES, KBError, KnowledgeBase, parse_recipe, resolve_allowed
from soundkb.store.schema import PatchRecipe

settings = load_settings()
app = FastAPI(title="SoundKB", docs_url=None, redoc_url=None)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@lru_cache(maxsize=1)
def _kb() -> KnowledgeBase:
    return KnowledgeBase(settings)


def kb() -> KnowledgeBase:
    """The shared service; a missing Qdrant becomes a readable 503."""
    try:
        return _kb()
    except KBError as exc:
        raise HTTPException(503, str(exc)) from exc


def entry_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise HTTPException(404, "Unknown entry.") from exc


def split_tags(raw: str) -> list[str]:
    return [t for t in re.split(r"[,\n]", raw) if t.strip()]


class EntryPatch(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    recipes: list[PatchRecipe] | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    kb()
    return {"ok": True, "allowed_dirs": [str(p) for p in settings.allowed_ingest_dirs]}


@app.post("/api/entries")
def create_entry(kind: str = Form("sound"), title: str = Form(""), tags: str = Form(""), notes: str = Form(""),
                 patch_recipe: str = Form(""), path: str = Form(""), allow_long: bool = Form(False),
                 file: UploadFile | None = File(None)) -> dict[str, Any]:
    """Create a sound (from an upload or an allowed local path), a note or a device map."""
    try:
        recipes = [parse_recipe(patch_recipe)] if patch_recipe.strip() else []
        if kind != "sound":
            if not title.strip():
                raise KBError("Title is required.")
            return kb().add_text(kind, title.strip(), notes, split_tags(tags), recipes)
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in AUDIO_SUFFIXES:
                raise KBError("Supported formats: WAV, AIFF, FLAC, MP3.")
            tmp = settings.data_dir / "tmp" / f"{uuid.uuid4().hex}{suffix}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            try:
                with tmp.open("wb") as out:
                    while chunk := file.file.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise KBError("File is larger than 100 MB.")
                        out.write(chunk)
                return kb().ingest_audio(tmp, title.strip() or Path(file.filename).stem, split_tags(tags), notes,
                                         recipes, allow_long, source_path=file.filename)
            finally:
                tmp.unlink(missing_ok=True)
        if path.strip():
            audio = resolve_allowed(path.strip(), settings)
            return kb().ingest_audio(audio, title.strip() or audio.stem, split_tags(tags), notes, recipes, allow_long)
        raise KBError("Choose an audio file or enter a local path.")
    except KBError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/entries")
def list_entries(q: str = "", tag: list[str] = Query(default=[]), kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return kb().search(q, kind=kind or None, tags=tag, limit=min(limit, 200))


@app.get("/api/entries/{point_id}")
def read_entry(point_id: str) -> dict[str, Any]:
    try:
        return kb().get(entry_id(point_id))
    except KBError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.patch("/api/entries/{point_id}")
def patch_entry(point_id: str, body: EntryPatch) -> dict[str, Any]:
    try:
        return kb().update(entry_id(point_id), body.title, body.tags, body.notes, body.recipes)
    except KBError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/entries/{point_id}")
def remove_entry(point_id: str) -> dict[str, bool]:
    try:
        kb().delete(entry_id(point_id))
    except KBError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": True}


@app.get("/api/entries/{point_id}/report.png")
def report_image(point_id: str) -> FileResponse:
    png = kb().report_png(entry_id(point_id))
    if not png or not png.is_file():
        raise HTTPException(404, "No report image.")
    return FileResponse(png, media_type="image/png")


@app.get("/api/entries/{point_id}/audio")
def audio(point_id: str) -> FileResponse:
    audio_path = kb().get(entry_id(point_id)).get("audio_path")
    if not audio_path or not Path(audio_path).is_file():
        raise HTTPException(404, "No audio.")
    return FileResponse(audio_path)


@app.get("/api/tags")
def tags() -> list[dict[str, Any]]:
    return kb().tags()


def main() -> None:
    """Start the interface at 127.0.0.1."""
    uvicorn.run(app, host="127.0.0.1", port=settings.ui_port)
