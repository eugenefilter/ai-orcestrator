"""Distance between a reference sound and a rendered candidate, with plain-text advice for the agent.

``distance`` is the mean of per-feature errors, each divided by a tolerance and capped at 3:
0 is identical, about 1 means "every feature is just off tolerance", 3 means "completely different".
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import librosa
import librosa.display
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
    for key, tol in (("attack_ms", .5), ("hold_ms", .5), ("decay_to_minus3_ms", .3), ("decay_to_minus6_ms", .25),
                     ("decay_to_minus10_ms", .25), ("decay_to_minus20_ms", .25), ("decay_to_minus40_ms", .3)):
        if re_[key] is not None and ce[key] is not None:
            base = max(re_[key], {"attack_ms": 20, "hold_ms": 50}.get(key, 1))  # tiny values would make the ratio explode
            rows[key] = _row(re_[key], ce[key], abs(ce[key] - re_[key]) / base, tol)
    rb, cb = ref["spectrum"]["band_energy_pct"], cand["spectrum"]["band_energy_pct"]
    for band in BANDS:
        rows[f"band_{band}_pct"] = _row(rb[band], cb[band], abs(cb[band] - rb[band]), 10)
    rh, ch = ref["spectrum"]["harmonics_db_relative"][:8], cand["spectrum"]["harmonics_db_relative"][:8]
    if rh and ch:
        pairs = [(a, b) for a, b in zip(rh, ch) if a is not None and b is not None]
        rows["harmonics_db"] = _row(None, None, float(np.mean([abs(b - a) for a, b in pairs])) if pairs else None, 6)
    rt, ct = ref["distortion"]["thd_pct"], cand["distortion"]["thd_pct"]
    if rt is not None and ct is not None:
        rows["thd_db"] = _row(rt, ct, abs(20 * math.log10(max(ct, .05) / max(rt, .05))), 6)
    ref_ev = {e["t_s"]: e for e in ref.get("evolution", [])}
    pairs_ev = [(ref_ev[e["t_s"]], e) for e in cand.get("evolution", []) if e["t_s"] in ref_ev]
    if pairs_ev:
        rows["evolution_centroid"] = _row(None, None, float(np.mean([_rel(a["centroid_hz"], b["centroid_hz"]) for a, b in pairs_ev])), .4)
        h3 = [abs(b["h3_db"] - a["h3_db"]) for a, b in pairs_ev if a["h3_db"] > -80 or b["h3_db"] > -80]
        rows["evolution_h3_db"] = _row(None, None, float(np.mean(h3)) if h3 else 0.0, 6)
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


def advice(per: dict[str, dict[str, Any]], harmonics: dict[str, list], level_diff_db: float | None = None) -> list[str]:
    """Rule-based, conservative hints for features that are off tolerance (norm > 1), most structural first."""
    out: list[str] = []
    calibrate = "Use solve_param (after calibrate_param) to turn a target time into a parameter value."

    def off(name: str) -> dict[str, Any] | None:
        row = per.get(name)
        return row if row and row["norm"] > 1 else None

    if (r := off("f0_steady_cents")) and r["ref"] and r["cand"]:
        cents = 1200 * math.log2(r["cand"] / r["ref"])
        semis = round(-cents / 100)
        if abs(cents) >= 150:
            out.append(f"Pitch is {abs(cents) / 100:.1f} semitones {'too high' if cents > 0 else 'too low'} ({r['cand']} Hz vs {r['ref']} Hz): change Transpose by {semis:+d} st first.")
        else:
            out.append(f"Pitch is {cents:+.0f} cents off ({r['cand']} Hz vs {r['ref']} Hz): detune by {-cents:+.0f} cents (Fine/Transpose).")
    if (r := off("pitch_drop_ms")):
        if r["cand"] is None:
            out.append(f"Candidate has no pitch drop; reference drops in {r['ref']} ms: add a pitch envelope. {calibrate}")
        elif r["ref"] is None:
            out.append("Candidate drops in pitch but the reference does not: reduce the pitch envelope amount.")
        else:
            out.append(f"Pitch drop takes {r['cand']} ms, reference {r['ref']} ms (x{r['ref'] / r['cand']:.2f}): {'shorten' if r['cand'] > r['ref'] else 'lengthen'} the pitch envelope decay. {calibrate}")
    if (r := off("pitch_drop_semitones")) and r["ref"] and r["cand"]:
        out.append(f"Pitch drop spans {r['cand']} st, reference {r['ref']} st: scale the pitch envelope amount by x{r['ref'] / r['cand']:.2f}.")
    if (r := off("attack_ms")):
        out.append(f"Attack is {r['cand']} ms, reference {r['ref']} ms (x{r['ref'] / max(r['cand'], 1):.2f}): {'shorten' if r['cand'] > r['ref'] else 'lengthen'} the amp attack. {calibrate}")
    if (r := off("hold_ms")):
        out.append(f"Level stays within 1 dB of the peak for {r['cand']} ms, reference {r['ref']} ms: the reference has a {'longer' if r['ref'] > r['cand'] else 'shorter'} flat top (slower initial decay, raise peak/sustain or lengthen decay).")
    for key, label in (("decay_to_minus3_ms", "-3 dB"), ("decay_to_minus6_ms", "-6 dB"), ("decay_to_minus10_ms", "-10 dB"),
                       ("decay_to_minus20_ms", "-20 dB"), ("decay_to_minus40_ms", "-40 dB")):
        if (r := off(key)):
            out.append(f"Decay to {label} is {r['cand']} ms, reference {r['ref']} ms (x{r['ref'] / r['cand']:.2f}): {'shorten' if r['cand'] > r['ref'] else 'lengthen'} the amp decay/release. {calibrate}")
            break
    for band in BANDS:
        if (r := off(f"band_{band}_pct")):
            out.append(f"{band} band holds {r['cand']}% of the energy, reference {r['ref']}%: {'cut' if r['cand'] > r['ref'] else 'boost'} that range (filter/EQ) or its oscillator level.")
    if (r := off("thd_db")):
        out.append(f"Harmonic distortion is {r['cand']}% vs {r['ref']}% in the reference: {'reduce drive/saturation' if r['cand'] > r['ref'] else 'add a Saturator (or Operator shaper/drive)'}.")
    elif off("harmonics_db"):
        pairs = [(a, b) for a, b in zip(harmonics["ref"], harmonics["cand"]) if a is not None and b is not None]
        diff = float(np.mean([b - a for a, b in pairs[1:]])) if len(pairs) > 1 else 0
        out.append(f"Harmonics 2-8 are {abs(diff):.0f} dB {'louder' if diff > 0 else 'quieter'} than the reference: {'reduce drive / lower the filter cutoff' if diff > 0 else 'add saturation/drive or open the filter'}.")
    if off("evolution_centroid") or off("evolution_h3_db"):
        out.append("Brightness over time differs (the reference darkens differently as it decays): check filter envelope/cutoff and drive after the level/time hints.")
    if (r := off("centroid_hz")):
        out.append(f"Brightness (centroid) is {r['cand']} Hz vs {r['ref']} Hz: {'darken' if r['cand'] > r['ref'] else 'brighten'} with the filter cutoff.")
    if level_diff_db is not None and abs(level_diff_db) > 3:
        out.append(f"Candidate is {abs(level_diff_db):.0f} dB {'louder' if level_diff_db > 0 else 'quieter'} (loudness is normalised in the distance; match it with track/device volume).")
    if (r := off("mel_l1_db")):
        out.append(f"Spectrogram differs by {r['error']} dB on average over time; check the earlier hints first.")
    return out


def plot_comparison(ref: tuple[np.ndarray, int, dict], cand: tuple[np.ndarray, int, dict], out_png: Path) -> None:
    """Side-by-side figure: envelope and f0 overlays, both spectrograms, harmonic bars (reference blue, candidate orange)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = fig.add_gridspec(4, 2)
    ax_env, ax_f0, ax_hm = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :]), fig.add_subplot(grid[3, :])
    for (y, sr, feats), color, name in ((ref, "tab:blue", "reference"), (cand, "tab:orange", "candidate")):
        env = feats["envelope"]
        ax_env.plot(env["env_times_s"], np.array(env["env_dbfs"], dtype=float) - max(env["env_dbfs"]), color=color, label=name)
        p = feats["pitch"]
        ax_f0.plot(p["times_s"], [np.nan if v is None else v for v in p["f0_hz"]], color=color, label=name)
    ax_env.set(title="Peak envelope (dB re peak)", ylim=(-60, 3), xlabel="Time (s)", ylabel="dB")
    ax_env.legend(loc="upper right")
    ax_f0.set(title="Fundamental frequency", xlabel="Time (s)", ylabel="Hz", yscale="log")
    ax_f0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_f0.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for col, (data, name) in enumerate(((ref, "Reference"), (cand, "Candidate"))):
        ax = fig.add_subplot(grid[2, col])
        fmax = min(16000, data[1] / 2)
        mel = librosa.feature.melspectrogram(y=data[0], sr=data[1], n_mels=96, fmin=20, fmax=fmax)
        librosa.display.specshow(librosa.power_to_db(mel, ref=1.0, top_db=None).clip(min=-80), sr=data[1], x_axis="time", y_axis="mel", fmin=20, fmax=fmax, ax=ax, vmin=-80, vmax=0, cmap="magma")
        ax.set(title=f"{name} spectrogram")
    width = .4
    for offset, (data, color) in enumerate(((ref, "tab:blue"), (cand, "tab:orange"))):
        h = [v if v is not None else -100 for v in data[2]["spectrum"]["harmonics_db_relative"]]
        ax_hm.bar(np.arange(1, len(h) + 1) + (offset - .5) * width, h, width, color=color)
    ax_hm.set(title="Harmonic profile (dB re fundamental)", xlabel="Harmonic", ylabel="dB")
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def compare_audio_files(reference: Path, candidate: Path, plot_to: Path | None = None) -> dict[str, Any]:
    """Full comparison: descriptor errors plus log-mel distance. Returns ``distance``, ``per_feature`` and ``advice``.

    With ``plot_to`` also writes a side-by-side PNG and returns its path as ``png_path``.
    """
    ry, rc, rsr = load_audio(reference)
    cy, cc, csr = load_audio(candidate)
    rf, cf = extract_features(ry, rc, rsr), extract_features(cy, cc, csr)
    result = compare_features(rf, cf)
    ra, ca = _prepare(ry, rsr, rsr), _prepare(cy, csr, rsr)
    size = min(len(ra), len(ca))
    mel = float(np.mean(np.abs(_log_mel(ra[:size], rsr) - _log_mel(ca[:size], rsr)))) if size > 2048 else None
    # a length mismatch also counts: the shorter clip is padded with silence
    result["per_feature"]["mel_l1_db"] = _row(None, None, mel, 6)
    per = result["per_feature"]
    distance = float(np.mean([r["norm"] for r in per.values()]))
    ref_lufs, cand_lufs = rf["meta"]["integrated_lufs"], cf["meta"]["integrated_lufs"]
    level_diff = None if ref_lufs is None or cand_lufs is None else round(cand_lufs - ref_lufs, 1)
    out = {"distance": round(distance, 3), "level_diff_db": level_diff, "per_feature": per,
           "advice": advice(per, result["harmonics"], level_diff)}
    if plot_to is not None:
        plot_to.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison((ry, rsr, rf), (cy, csr, cf), plot_to)
        out["png_path"] = str(plot_to)
    return out
