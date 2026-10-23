"""MCP server (stdio): lets an agent search the sound knowledge base, analyze and compare audio, save recipes."""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import json
import time
from pathlib import Path

from soundkb.analysis.compare import compare_audio_files
from soundkb.autotune import ParamSpec, optimize
from soundkb.bridge import AbletonBridge, BridgeError
from soundkb.calibrate import calibrate, calibration_path, solve
from soundkb.record import RecordError, record_clip
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
        except (KBError, ValueError, BridgeError, RecordError) as exc:
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
    """Compare a rendered candidate with a stored sound (reference_id) or a file (reference_path). Returns distance (0 identical, ~1 every feature just off tolerance, 3 completely different), per_feature errors and plain-text advice on which synth parameter to change. Also returns png_path: a side-by-side figure (envelope, f0, spectrograms, harmonics) to open with your image reader. Lower distance after an edit means the patch got closer."""
    if reference_id:
        reference = _kb().get(reference_id).get("audio_path")
        if not reference:
            raise KBError("That entry has no audio.")
        ref = Path(reference)
    elif reference_path:
        ref = resolve_allowed(reference_path, settings)
    else:
        raise KBError("Give reference_id or reference_path.")
    png = settings.data_dir / "comparisons" / f"{int(time.time())}.png"
    return compare_audio_files(ref, resolve_allowed(candidate_path, settings), png)


def _bridge() -> AbletonBridge:
    return AbletonBridge(settings.ableton_port)


def _renderer(track_index: int, clip_index: int, seconds: float):
    def render(out: Path) -> dict:
        return record_clip(_bridge(), track_index, clip_index, seconds, out, settings.audio_input)
    return render


@mcp.tool()
@_guard
def record_render(track_index: int, clip_index: int, seconds: float = 4.0, name: str = "render") -> dict[str, Any]:
    """Fire a clip in Ableton and record its output to a WAV (Live's output must be routed to the BlackHole device, see README). Leading silence is trimmed. Returns path, peak_dbfs and duration; pass the path to compare_audio or analyze_audio. Look the track up by name first: indices change."""
    out = settings.data_dir / "renders" / f"{name}_{int(time.time())}.wav"
    return _renderer(track_index, clip_index, seconds)(out)


@mcp.tool()
@_guard
def calibrate_param(track_index: int, device_index: int, parameter_index: int, values: list[float], clip_index: int = 0, seconds: float = 4.0) -> dict[str, Any]:
    """Learn a parameter's real scale: set each value, render a note, measure attack/hold/decay/pitch-drop/f0/brightness/THD, restore the original value and save the table. Use 4-6 values spread over the range (e.g. 0.2, 0.4, 0.6, 0.8). Afterwards solve_param converts a target ("attack 75 ms") into a value. Takes ~seconds+1 s per value."""
    result = calibrate(_bridge(), _renderer(track_index, clip_index, seconds), track_index, device_index, parameter_index, values,
                       settings.data_dir / "calibration", settings.data_dir / "renders")
    table = [{"value": p["value"], **{k: v for k, v in p["metrics"].items() if v is not None}} for p in result["points"]]
    _kb().add_text("device_map", f"Calibration: {result['device']} / {result['parameter']}", json.dumps(table), ["calibration", result["device"].lower()])
    return {"device": result["device"], "parameter": result["parameter"], "points": table}


@mcp.tool()
@_guard
def solve_param(device: str, parameter: str, metric: str, target: float) -> dict[str, Any]:
    """Parameter value that should give metric == target, from a saved calibration. metric: attack_ms, hold_ms, decay_to_minus{3,6,10,20,40}_ms, pitch_drop_ms, f0_steady, centroid_hz, thd_pct, sub_pct. If extrapolated is true, calibrate again with values around the result."""
    path = calibration_path(settings.data_dir / "calibration", device, parameter)
    if not path.is_file():
        raise KBError(f"No calibration for {device} / {parameter}. Run calibrate_param first.")
    return solve(json.loads(path.read_text()), metric, target)


@mcp.tool()
@_guard
def optimize_patch(track_index: int, device_index: int, clip_index: int, params: list[dict[str, float]], reference_id: str | None = None,
                   reference_path: str | None = None, max_renders: int = 20, seconds: float = 4.0) -> dict[str, Any]:
    """Automatic coordinate-descent tuning: probes each parameter at +-step, renders, keeps what lowers compare distance, halves steps. params: [{index, lo, hi, step}] (current values are read from the device). Each render takes ~seconds+1 s, so keep max_renders small and choose only parameters the compare advice points to. The best values are left set on the device; returns them with the distance history."""
    if reference_id:
        ref_audio = _kb().get(reference_id).get("audio_path")
        if not ref_audio:
            raise KBError("That entry has no audio.")
        ref = Path(ref_audio)
    elif reference_path:
        ref = resolve_allowed(reference_path, settings)
    else:
        raise KBError("Give reference_id or reference_path.")
    bridge = _bridge()
    current = {p["index"]: p for p in bridge.device_parameters(track_index, device_index)}
    specs = [ParamSpec(int(p["index"]), p["lo"], p["hi"], current[int(p["index"])]["value"], p["step"], current[int(p["index"])]["name"]) for p in params]
    render = _renderer(track_index, clip_index, seconds)
    count = {"n": 0}

    def evaluate(values: dict[int, float]) -> float:
        for index, value in values.items():
            bridge.set_parameter(track_index, device_index, index, value)
        count["n"] += 1
        out = settings.data_dir / "renders" / f"opt_{int(time.time())}_{count['n']}.wav"
        render(out)
        return compare_audio_files(ref, out)["distance"]

    result = optimize(specs, evaluate, max_renders)
    for index, value in result["values"].items():
        bridge.set_parameter(track_index, device_index, index, value)
    names = {s.index: s.name for s in specs}
    return {"distance": result["distance"], "renders": result["renders"], "values": {f"{names[i]} ({i})": v for i, v in result["values"].items()},
            "history": [{"distance": round(h["distance"], 3)} for h in result["history"]]}


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
