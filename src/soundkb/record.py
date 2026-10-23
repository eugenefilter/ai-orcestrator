"""Capture a rendered note: fire a Live clip and record Live's output (routed to a virtual device such as BlackHole)."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .bridge import AbletonBridge

SILENCE = 1e-3  # -60 dBFS: leading samples below this are trimmed so the onset sits at t=0


class RecordError(RuntimeError):
    """Recording failed or captured silence."""


def parse_audio_devices(ffmpeg_listing: str) -> dict[str, int]:
    """Parse ``ffmpeg -f avfoundation -list_devices`` output into ``{device name: index}`` (audio section only)."""
    devices: dict[str, int] = {}
    in_audio = False
    for line in ffmpeg_listing.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
        elif "AVFoundation video devices" in line:
            in_audio = False
        elif in_audio and (m := re.search(r"\[(\d+)\]\s+(.+)$", line)):
            devices[m.group(2).strip()] = int(m.group(1))
    return devices


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RecordError("ffmpeg is not installed (brew install ffmpeg).")
    return path


def list_audio_devices() -> dict[str, int]:
    out = subprocess.run([_ffmpeg(), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True, timeout=20)
    return parse_audio_devices(out.stderr)


def trim_leading_silence(samples: np.ndarray, threshold: float = SILENCE) -> tuple[np.ndarray, int]:
    """Drop samples before the first one above ``threshold``; returns the trimmed array and the number removed."""
    above = np.flatnonzero(np.max(np.abs(samples), axis=1) > threshold) if samples.ndim > 1 else np.flatnonzero(np.abs(samples) > threshold)
    start = int(above[0]) if above.size else 0
    return samples[start:], start


def record_clip(bridge: AbletonBridge, track_index: int, clip_index: int, seconds: float, out_path: Path,
                device: str = "BlackHole 2ch", pre_roll: float = 0.6) -> dict[str, float | str]:
    """Record ``seconds`` of audio while the clip plays; trim leading silence and save a WAV at ``out_path``.

    Live's output must be routed to ``device`` (see README). The transport is stopped first so the clip launches
    immediately instead of waiting for launch quantization.
    """
    devices = list_audio_devices()
    if device not in devices:
        raise RecordError(f"Audio input '{device}' not found. Available: {', '.join(devices) or 'none'}.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.wav")
    bridge.send("stop_playback")
    capture = subprocess.Popen([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", f":{devices[device]}",
                                "-t", f"{seconds + pre_roll:.2f}", "-ar", "44100", "-c:a", "pcm_f32le", str(raw)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        time.sleep(pre_roll)
        bridge.send("fire_clip", track_index=track_index, clip_index=clip_index)
        _, err = capture.communicate(timeout=seconds + 15)
    finally:
        if capture.poll() is None:
            capture.kill()
        try:
            bridge.send("stop_clip", track_index=track_index, clip_index=clip_index)
        except Exception:
            pass
    if capture.returncode != 0 or not raw.exists():
        raise RecordError(f"ffmpeg failed: {err.decode(errors='replace')[:300] if err else 'no output'}")
    samples, sr = sf.read(raw, dtype="float32", always_2d=True)
    raw.unlink(missing_ok=True)
    trimmed, removed = trim_leading_silence(samples)
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    if peak < SILENCE:
        raise RecordError(f"Captured silence from '{device}'. Route Live's audio output to it (Preferences > Audio > Output Device, "
                          "or a Multi-Output Device that includes it) and make sure the clip is not muted.")
    sf.write(out_path, trimmed, sr)
    return {"path": str(out_path), "peak_dbfs": round(20 * float(np.log10(peak)), 1), "duration_s": round(len(trimmed) / sr, 3),
            "onset_offset_s": round(removed / sr, 3)}
