"""Distance between a reference sound and a rendered candidate, with plain-text advice for the agent.

``distance`` is the mean of per-feature errors, each divided by a tolerance and capped at 3:
0 is identical, about 1 means "every feature is just off tolerance", 3 means "completely different".
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .features import extract_features
from .load import load_audio

CAP = 3.0
BANDS = ("sub", "bass", "low_mid", "high_mid")


def _rel(ref: float, cand: float) -> float:
    return abs(cand - ref) / max(abs(ref), 1e-9)


def _row(ref: Any, cand: Any, error: float | None, tolerance: float) -> dict[str, Any]:
    norm = CAP if error is None else min(error / tolerance, CAP)
    return {"ref": ref, "cand": cand, "error": None if error is None else round(error, 3), "tolerance": tolerance, "norm": round(norm, 3)}


def compare_features(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """Compare two ``extract_features`` results (scalar descriptors only; no audio needed)."""
    rows: dict[str, dict[str, Any]] = {}
    rf, cf = ref["pitch"]["f0_steady"], cand["pitch"]["f0_steady"]
    if rf or cf:
        cents = abs(1200 * math.log2(cf / rf)) if rf and cf else None
        rows["f0_steady_cents"] = _row(rf, cf, cents, 30)
    rd, cd = ref["pitch"]["pitch_drop"], cand["pitch"]["pitch_drop"]
    if rd or cd:
        rows["pitch_drop_ms"] = _row(rd and rd["time_to_90pct_ms"], cd and cd["time_to_90pct_ms"],
                                     _rel(rd["time_to_90pct_ms"], cd["time_to_90pct_ms"]) if rd and cd else None, .3)
        rows["pitch_drop_semitones"] = _row(rd and round(12 * math.log2(rd["start_hz"] / rd["end_hz"]), 2),
                                            cd and round(12 * math.log2(cd["start_hz"] / cd["end_hz"]), 2),
                                            abs(12 * math.log2(rd["start_hz"] / rd["end_hz"]) - 12 * math.log2(cd["start_hz"] / cd["end_hz"])) if rd and cd else None, 1.5)
    re_, ce = ref["envelope"], cand["envelope"]
    for key, tol in (("attack_ms", .5), ("decay_to_minus6_ms", .25), ("decay_to_minus20_ms", .25), ("decay_to_minus40_ms", .3)):
        if re_[key] is not None and ce[key] is not None:
            base = max(re_[key], 20) if key == "attack_ms" else re_[key]  # tiny attacks would make the ratio explode
            rows[key] = _row(re_[key], ce[key], abs(ce[key] - re_[key]) / base, tol)
    rb, cb = ref["spectrum"]["band_energy_pct"], cand["spectrum"]["band_energy_pct"]
    for band in BANDS:
        rows[f"band_{band}_pct"] = _row(rb[band], cb[band], abs(cb[band] - rb[band]), 10)
    rh, ch = ref["spectrum"]["harmonics_db_relative"][:8], cand["spectrum"]["harmonics_db_relative"][:8]
    if rh and ch:
        pairs = [(a, b) for a, b in zip(rh, ch) if a is not None and b is not None]
        rows["harmonics_db"] = _row(None, None, float(np.mean([abs(b - a) for a, b in pairs])) if pairs else None, 6)
    rows["centroid_hz"] = _row(ref["spectrum"]["centroid_hz"], cand["spectrum"]["centroid_hz"],
                               _rel(ref["spectrum"]["centroid_hz"], cand["spectrum"]["centroid_hz"]), .3)
    return {"per_feature": rows, "harmonics": {"ref": rh, "cand": ch}}


def _log_mel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64, fmin=20, fmax=min(16000, sr / 2))
    return librosa.power_to_db(mel, ref=1.0, top_db=None).clip(min=-80)


def _prepare(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Trim leading silence (onset alignment), resample and normalize RMS."""
    above = np.flatnonzero(np.abs(y) > max(float(np.max(np.abs(y))) * .02, 1e-6))
    y = y[above[0]:] if above.size else y
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y / max(float(np.sqrt(np.mean(y**2))), 1e-9) * .1


def advice(per: dict[str, dict[str, Any]], harmonics: dict[str, list]) -> list[str]:
    """Rule-based, conservative hints for features that are off tolerance (norm > 1)."""
    out: list[str] = []

    def off(name: str) -> dict[str, Any] | None:
        row = per.get(name)
        return row if row and row["norm"] > 1 else None

    if (r := off("f0_steady_cents")) and r["ref"] and r["cand"]:
        out.append(f"Pitch is {1200 * math.log2(r['cand'] / r['ref']):+.0f} cents off ({r['cand']} Hz vs {r['ref']} Hz): retune the oscillator (Transpose/Fine).")
    if (r := off("pitch_drop_ms")):
        if r["cand"] is None:
            out.append(f"Candidate has no pitch drop; reference drops in {r['ref']} ms: add a pitch envelope.")
        elif r["ref"] is None:
            out.append("Candidate drops in pitch but the reference does not: reduce the pitch envelope amount.")
        else:
            out.append(f"Pitch drop takes {r['cand']} ms, reference {r['ref']} ms: {'shorten' if r['cand'] > r['ref'] else 'lengthen'} the pitch envelope decay.")
    if (r := off("pitch_drop_semitones")) and r["ref"] and r["cand"]:
        out.append(f"Pitch drop spans {r['cand']} st, reference {r['ref']} st: {'reduce' if r['cand'] > r['ref'] else 'increase'} the pitch envelope amount.")
    if (r := off("attack_ms")):
        out.append(f"Attack is {r['cand']} ms, reference {r['ref']} ms: {'shorten' if r['cand'] > r['ref'] else 'lengthen'} the amp attack.")
    for key, label in (("decay_to_minus6_ms", "-6 dB"), ("decay_to_minus20_ms", "-20 dB"), ("decay_to_minus40_ms", "-40 dB")):
        if (r := off(key)):
            out.append(f"Decay to {label} is {r['cand']} ms, reference {r['ref']} ms: {'shorten' if r['cand'] > r['ref'] else 'lengthen'} the amp decay/release.")
            break
    for band in BANDS:
        if (r := off(f"band_{band}_pct")):
            out.append(f"{band} band holds {r['cand']}% of the energy, reference {r['ref']}%: {'cut' if r['cand'] > r['ref'] else 'boost'} that range (filter/EQ) or its oscillator level.")
    if off("harmonics_db"):
        pairs = [(a, b) for a, b in zip(harmonics["ref"], harmonics["cand"]) if a is not None and b is not None]
        diff = float(np.mean([b - a for a, b in pairs[1:]])) if len(pairs) > 1 else 0
        out.append(f"Harmonics 2-8 are {abs(diff):.0f} dB {'louder' if diff > 0 else 'quieter'} than the reference: {'reduce drive / lower the filter cutoff' if diff > 0 else 'add saturation/drive or open the filter'}.")
    if (r := off("centroid_hz")):
        out.append(f"Brightness (centroid) is {r['cand']} Hz vs {r['ref']} Hz: {'darken' if r['cand'] > r['ref'] else 'brighten'} with the filter cutoff.")
    if (r := off("mel_l1_db")):
        out.append(f"Spectrogram differs by {r['error']} dB on average over time; check the earlier hints first.")
    return out


def compare_audio_files(reference: Path, candidate: Path) -> dict[str, Any]:
    """Full comparison: descriptor errors plus log-mel distance. Returns ``distance``, ``per_feature`` and ``advice``."""
    ry, rc, rsr = load_audio(reference)
    cy, cc, csr = load_audio(candidate)
    result = compare_features(extract_features(ry, rc, rsr), extract_features(cy, cc, csr))
    ra, ca = _prepare(ry, rsr, rsr), _prepare(cy, csr, rsr)
    size = min(len(ra), len(ca))
    mel = float(np.mean(np.abs(_log_mel(ra[:size], rsr) - _log_mel(ca[:size], rsr)))) if size > 2048 else None
    # a length mismatch also counts: the shorter clip is padded with silence
    result["per_feature"]["mel_l1_db"] = _row(None, None, mel, 6)
    per = result["per_feature"]
    distance = float(np.mean([r["norm"] for r in per.values()]))
    return {"distance": round(distance, 3), "per_feature": per, "advice": advice(per, result["harmonics"])}
