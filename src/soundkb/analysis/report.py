"""Analysis report assembly and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .features import extract_features
from .load import load_audio

CURVE_POINTS = 64


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g} ms"


def build_summary(features: dict[str, Any]) -> list[str]:
    """Generate concise English conclusions from numerical features."""
    meta, pitch, env, spectrum = features["meta"], features["pitch"], features["envelope"], features["spectrum"]
    lines = [f"Duration {meta['duration_s']} s at {meta['sr']} Hz; peak {meta['peak_dbfs']} dBFS and RMS {meta['rms_dbfs']} dBFS."]
    sub = spectrum["band_energy_pct"]["sub"]
    lines.append(f"{'Sub-heavy' if sub >= 40 else 'Broadband'}: {sub}% energy below 60 Hz.")
    if pitch["f0_steady"]:
        lines.append(f"Steady pitch {pitch['f0_steady']} Hz ({pitch['note']}, {pitch['cents']} cents).")
    if pitch["pitch_drop"]:
        lines.append("f0 drops from {start_hz} Hz to {end_hz} Hz, 90% of the way by {time_to_90pct_ms} ms.".format(**pitch["pitch_drop"]))
    lines.append(f"Attack (10% to -3 dB) {_ms(env['attack_ms'])}, peak at {_ms(env['peak_time_ms'])}; decay to -40 dB at {_ms(env['decay_to_minus40_ms'])}.")
    harmonics = spectrum["harmonics_db_relative"]
    if len(harmonics) >= 3:
        lines.append(f"Second/third harmonic {harmonics[1]} / {harmonics[2]} dB relative to the fundamental.")
    lines.append("Likely one-shot." if env["is_one_shot"] else "Sustained or evolving sound.")
    return lines


def _downsample(times: list[float], values: list[float | None]) -> list[list[float | None]]:
    """Pick at most ``CURVE_POINTS`` evenly spaced (time, value) pairs."""
    if not times:
        return []
    ids = np.unique(np.linspace(0, len(times) - 1, min(CURVE_POINTS, len(times))).astype(int))
    return [[times[i], values[i]] for i in ids]


def compact(features: dict[str, Any]) -> dict[str, Any]:
    """Replace the full f0 and envelope arrays with short curves so the JSON stays small for agents."""
    pitch = {k: v for k, v in features["pitch"].items() if k not in ("f0_hz", "times_s", "voiced")}
    pitch["f0_curve_hz"] = _downsample(features["pitch"]["times_s"], features["pitch"]["f0_hz"])
    env = {k: v for k, v in features["envelope"].items() if k not in ("env_times_s", "env_dbfs")}
    env["env_curve_dbfs"] = _downsample(features["envelope"]["env_times_s"], features["envelope"]["env_dbfs"])
    return {**features, "pitch": pitch, "envelope": env}


def _analyze(path: Path) -> tuple[dict[str, Any], np.ndarray, int]:
    y, channels, sr = load_audio(path)
    return extract_features(y, channels, sr), y, sr


def _report(features: dict[str, Any]) -> dict[str, Any]:
    return {"schema": 1, "summary": build_summary(features), "features": compact(features)}


def analyze_file(path: Path) -> dict[str, Any]:
    """Analyze an audio file and return the versioned, compact report."""
    features, _, _ = _analyze(path)
    return _report(features)


def write_report(path: Path, out_dir: Path) -> dict[str, Any]:
    """Write ``report.json`` and a 1600px-wide diagnostic PNG, returning the report."""
    features, y, sr = _analyze(path)
    report = _report(features)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), constrained_layout=True)
    axes[0].plot(np.arange(len(y)) / sr, y, lw=.45, label="waveform")
    env_t = features["envelope"]["env_times_s"]
    env = 10 ** (np.array(features["envelope"]["env_dbfs"], dtype=float) / 20)
    axes[0].plot(env_t, env, color="tab:red", lw=1, label="peak envelope")
    axes[0].plot(env_t, -env, color="tab:red", lw=1)
    axes[0].set(title="Waveform and peak envelope", xlabel="Time (s)", ylabel="Amplitude")
    axes[0].legend(loc="upper right")

    fmax = min(20000, sr / 2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=20, fmax=fmax)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, x_axis="time", y_axis="mel", fmin=20, fmax=fmax, ax=axes[1])
    axes[1].set(title="Mel spectrogram (dB)")

    p = features["pitch"]
    axes[2].plot(p["times_s"], p["f0_hz"], ".-", ms=2)
    axes[2].set(title="Fundamental frequency (frames >50 dB below peak are dropped)", xlabel="Time (s)", ylabel="Hz", yscale="log")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[2].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    h = features["spectrum"]["harmonics_db_relative"]
    axes[3].bar(range(1, len(h) + 1), [v if v is not None else -100 for v in h])
    axes[3].set(title="Harmonic profile", xlabel="Harmonic", ylabel="dB relative to fundamental")
    fig.savefig(out_dir / "report.png", dpi=100)
    plt.close(fig)
    return report
