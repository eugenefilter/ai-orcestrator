"""Parameter calibration: sweep one device parameter, render each value, and record how the sound changes.

Many Ableton parameters are normalised 0..1 with a non-linear scale (Operator times, filter frequencies). The sweep
turns that unknown scale into measured points, so a target ("attack 75 ms") can be converted to a parameter value.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from .analysis.features import extract_features
from .analysis.load import load_audio
from .bridge import AbletonBridge

METRICS = {
    "attack_ms": ("envelope", "attack_ms"), "hold_ms": ("envelope", "hold_ms"),
    "decay_to_minus3_ms": ("envelope", "decay_to_minus3_ms"), "decay_to_minus6_ms": ("envelope", "decay_to_minus6_ms"),
    "decay_to_minus10_ms": ("envelope", "decay_to_minus10_ms"), "decay_to_minus20_ms": ("envelope", "decay_to_minus20_ms"),
    "decay_to_minus40_ms": ("envelope", "decay_to_minus40_ms"), "f0_steady": ("pitch", "f0_steady"),
    "pitch_drop_ms": ("pitch", "pitch_drop.time_to_90pct_ms"), "centroid_hz": ("spectrum", "centroid_hz"),
    "thd_pct": ("distortion", "thd_pct"), "peak_dbfs": ("meta", "peak_dbfs"), "sub_pct": ("spectrum", "band_energy_pct.sub"),
}


def measure(path: Path) -> dict[str, float | None]:
    """Extract the calibration metrics from a rendered file."""
    y, channels, sr = load_audio(path)
    features = extract_features(y, channels, sr)
    out: dict[str, float | None] = {}
    for name, (group, dotted) in METRICS.items():
        value: Any = features[group]
        for key in dotted.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        out[name] = value
    return out


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def calibration_path(store_dir: Path, device: str, parameter: str) -> Path:
    return store_dir / f"{slug(device)}__{slug(parameter)}.json"


def calibrate(bridge: AbletonBridge, render: Callable[[Path], dict], track_index: int, device_index: int, parameter_index: int,
              values: list[float], store_dir: Path, work_dir: Path) -> dict[str, Any]:
    """Render one note per value, measure it, restore the parameter and save ``<device>__<parameter>.json``.

    ``render(out_path)`` must fire the clip and write a WAV (see ``record.record_clip``).
    """
    params = bridge.device_parameters(track_index, device_index)
    spec = next(p for p in params if p["index"] == parameter_index)
    device = bridge.send("get_device_parameters", track_index=track_index, device_index=device_index)
    device_name = (device["device"]["name"] if isinstance(device, dict) else json.loads(device)["device"]["name"])
    original = spec["value"]
    points = []
    try:
        for value in values:
            bridge.set_parameter(track_index, device_index, parameter_index, value)
            wav = work_dir / f"calib_{slug(device_name)}_{slug(spec['name'])}_{value:g}.wav"
            render(wav)
            points.append({"value": value, "metrics": measure(wav)})
    finally:
        bridge.set_parameter(track_index, device_index, parameter_index, original)
    result = {"device": device_name, "parameter": spec["name"], "index": parameter_index, "min": spec["min"], "max": spec["max"],
              "restored_to": original, "points": points}
    path = calibration_path(store_dir, device_name, spec["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def solve(calibration: dict[str, Any], metric: str, target: float) -> dict[str, Any]:
    """Parameter value expected to give ``metric == target``: log-linear interpolation between measured points.

    Outside the measured range the nearest two points are extrapolated and ``extrapolated`` is true; then re-calibrate
    around the result for a trustworthy value.
    """
    pts = sorted(((p["value"], p["metrics"].get(metric)) for p in calibration["points"] if p["metrics"].get(metric) not in (None, 0)))
    if len(pts) < 2:
        raise ValueError(f"Need at least two measured points for '{metric}' (have {len(pts)}); re-run calibrate_param with more values.")
    use_log = all(m > 0 for _, m in pts) and target > 0
    f = (lambda v: math.log(v)) if use_log else (lambda v: v)
    xs, ys = [v for v, _ in pts], [f(m) for _, m in pts]
    t = f(target)
    # pick the bracketing segment, or the nearest end segment for extrapolation
    seg = next((i for i in range(len(xs) - 1) if min(ys[i], ys[i + 1]) <= t <= max(ys[i], ys[i + 1])), None)
    extrapolated = seg is None
    if seg is None:
        seg = 0 if abs(t - ys[0]) < abs(t - ys[-1]) else len(xs) - 2
    dy = ys[seg + 1] - ys[seg]
    if dy == 0:
        raise ValueError(f"'{metric}' does not change between {xs[seg]} and {xs[seg + 1]}; the parameter does not control it.")
    value = xs[seg] + (t - ys[seg]) * (xs[seg + 1] - xs[seg]) / dy
    value = min(max(value, calibration["min"]), calibration["max"])
    return {"value": round(value, 4), "extrapolated": extrapolated, "between": [xs[seg], xs[seg + 1]],
            "measured": [pts[seg][1], pts[seg + 1][1]]}
