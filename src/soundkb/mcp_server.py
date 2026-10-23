"""MCP server (stdio): lets an agent search the sound knowledge base, analyze and compare audio, save recipes."""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from soundkb.analysis.compare import compare_audio_files
from soundkb.analysis.report import write_report
from soundkb.config import load_settings
from soundkb.kb import KBError, KnowledgeBase, resolve_allowed, sha256, validate_recipe
from soundkb.store.schema import PatchParam

mcp = MCPServer("soundkb")
settings = load_settings()


@lru_cache(maxsize=1)
def _kb() -> KnowledgeBase:
    return KnowledgeBase(settings)


def _guard(fn):
    """Turn user-facing errors into ToolError, the only kind whose message reaches the agent."""
    @functools.wraps(fn)  # keeps the signature, which MCP turns into the tool's input schema
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KBError, ValueError) as exc:
            raise ToolError(str(exc)) from None
    return wrapper


def _slim(entry: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky curves; keep scalar features, the summary and checkpoints."""
    analysis = entry.get("analysis")
    if analysis:
        features = {k: dict(v) for k, v in analysis["features"].items()}
        features["pitch"].pop("f0_curve_hz", None)
        features["envelope"].pop("env_curve_dbfs", None)
        entry = {**entry, "analysis": {**analysis, "features": features}}
    return entry


@mcp.tool()
@_guard
def search_knowledge(query: str, kind: str | None = None, tags: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Semantic search over sounds, notes and device maps. kind: sound | note | device_map. Returns id, title, tags, analysis summary, notes and recipe names."""
    return _kb().search(query, kind=kind, tags=tags, limit=limit)


@mcp.tool()
@_guard
def get_entry(entry_id: str) -> dict[str, Any]:
    """Full entry: analysis (summary, pitch checkpoints, envelope times, band energies, harmonics), notes and all patch recipes. report.png path is under data/analysis/<sha>/."""
    return _slim(_kb().get(entry_id))


@mcp.tool()
@_guard
def analyze_audio(path: str) -> dict[str, Any]:
    """Analyze any allowed local audio file without storing it. Returns the summary and features, plus png_path: open that PNG with your image Read tool to see the spectrogram."""
    audio = resolve_allowed(path, settings)
    out = settings.data_dir / "analysis_tmp" / sha256(audio)
    report = write_report(audio, out)
    return _slim({"analysis": report})["analysis"] | {"png_path": str(out / "report.png")}


@mcp.tool()
@_guard
def ingest_audio(path: str, title: str, tags: list[str] | None = None, notes: str = "") -> dict[str, Any]:
    """Store an allowed local audio file as a 'sound' entry (analysis, tags, notes). Same file content updates the same entry."""
    entry = _kb().ingest_audio(resolve_allowed(path, settings), title, tags, notes)
    return {"id": entry["id"], "summary": entry["analysis"]["summary"]}


@mcp.tool()
@_guard
def compare_audio(candidate_path: str, reference_id: str | None = None, reference_path: str | None = None) -> dict[str, Any]:
    """Compare a rendered candidate with a stored sound (reference_id) or a file (reference_path). Returns distance (0 identical, ~1 every feature just off tolerance, 3 completely different), per_feature errors and plain-text advice on which synth parameter to change. Lower distance after an edit means the patch got closer."""
    if reference_id:
        reference = _kb().get(reference_id).get("audio_path")
        if not reference:
            raise KBError("That entry has no audio.")
        from pathlib import Path
        ref = Path(reference)
    elif reference_path:
        ref = resolve_allowed(reference_path, settings)
    else:
        raise KBError("Give reference_id or reference_path.")
    return compare_audio_files(ref, resolve_allowed(candidate_path, settings))


@mcp.tool()
@_guard
def save_patch_recipe(entry_id: str, recipe: dict[str, Any]) -> dict[str, Any]:
    """Attach a patch recipe to an entry (replaces one with the same name). Schema: name, target{device, browser_uri, live_version}, params[{name, index, value, value_display}], chain[{device, params}], midi{note, velocity, length_beats}, verified{distance, iterations, date, render_path}, confidence guess|close|verified. Store parameter values exactly as get_device_parameters returned them."""
    entry = _kb().add_recipe(entry_id, validate_recipe(recipe))
    return {"id": entry["id"], "recipes": [r["name"] for r in entry["patch_recipes"]]}


@mcp.tool()
@_guard
def save_note(title: str, text: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Store free-form knowledge (synthesis know-how, mixing tip, what worked) as a 'note' entry."""
    return {"id": _kb().add_text("note", title, text, tags)["id"]}


@mcp.tool()
@_guard
def save_device_map(device: str, parameters: list[dict[str, Any]], notes: str = "") -> dict[str, Any]:
    """Store what you read from get_device_parameters for one device (name, index, min/max, display, conversion, gotchas) as a 'device_map' entry. Only save values you actually read; never from memory."""
    for p in parameters:
        PatchParam.model_validate({"value": p.get("value", 0), **{k: v for k, v in p.items() if k in PatchParam.model_fields}})
    lines = [f"{p.get('name')} (index {p.get('index')}): min {p.get('min')} max {p.get('max')} display {p.get('value_display')} {p.get('note', '')}" for p in parameters]
    text = "\n".join(lines) + (f"\n{notes}" if notes else "")
    return {"id": _kb().add_text("device_map", f"Device map: {device}", text, ["device-map", device.lower()])["id"]}


@mcp.tool()
@_guard
def list_tags() -> list[dict[str, Any]]:
    """All tags with counts."""
    return _kb().tags()


def main() -> None:
    """Run over stdio."""
    mcp.run()
