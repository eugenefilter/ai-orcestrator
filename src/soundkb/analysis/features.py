"""Pure, deterministic feature extractors. All timings are seconds unless named ``_ms``."""

from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert


def _db(value: float) -> float:
    return float(20 * np.log10(max(abs(value), 1e-12)))


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def meta_features(y: np.ndarray, channels: np.ndarray, sr: int) -> dict[str, Any]:
    """Return duration, level (dBFS/LUFS), stereo width and silence metadata."""
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    threshold = max(peak * 0.001, 1e-6)
    leading = int(np.argmax(np.abs(y) > threshold)) / sr if np.any(np.abs(y) > threshold) else len(y) / sr
    loudness = None
    if len(y) / sr >= 0.4 and rms > 1e-8:
        try:
            loudness = float(pyln.Meter(sr).integrated_loudness(y.astype(np.float64)))
        except ValueError:
            pass
    side_ratio = 0.0
    if channels.shape[1] >= 2:
        mid = (channels[:, 0] + channels[:, 1]) / 2
        side = (channels[:, 0] - channels[:, 1]) / 2
        side_ratio = float(np.mean(side**2) / max(np.mean(mid**2), 1e-12))
    return {"duration_s": _round(len(y) / sr), "sr": sr, "channels": int(channels.shape[1]),
            "peak_dbfs": _round(_db(peak)), "rms_dbfs": _round(_db(rms)), "integrated_lufs": _round(loudness),
            "stereo_width_side_mid": _round(side_ratio), "dc_offset": _round(float(np.mean(y)), 6),
            "leading_silence_s": _round(leading)}


def pitch_features(y: np.ndarray, sr: int, fmin: float = 20, fmax: float = 2000) -> dict[str, Any]:
    """Estimate f0 in Hz with pYIN and summarize the early and steady pitch.

    Frames more than 50 dB below the peak level are discarded: pYIN returns noise-floor garbage once a
    sound has decayed. The analysis window is about 3 periods of ``fmin``, so ``f0_start`` is the first
    voiced frame (``f0_start_time_ms``), not the pitch at t=0, and drops faster than ~50 ms are blurred.
    """
    if np.max(np.abs(y)) < 1e-7 or len(y) < max(128, int(sr / fmin * 4)):
        return {"f0_hz": [], "times_s": [], "voiced": [], "f0_start": None, "f0_start_time_ms": None, "f0_at": {},
                "f0_steady": None, "note": None, "cents": None, "pitch_drop": None}
    frame = int(2 ** math.ceil(math.log2(max(2048, 3 * sr / fmin))))
    frame = min(frame, max(2048, 2 ** int(math.floor(math.log2(len(y))))))
    hop = max(64, min(256, frame // 4))
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    level = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][: len(f0)]
    level = np.pad(level, (0, len(f0) - len(level)))
    f0 = np.where(level >= level.max() * 10 ** (-50 / 20), f0, np.nan)
    valid = np.isfinite(f0)

    def median_window(start: float, end: float) -> float | None:
        values = f0[valid & (times >= start) & (times <= end)]
        return float(np.median(values)) if values.size else None

    duration = len(y) / sr
    audible = float(times[valid][-1]) if valid.any() else duration
    steady = median_window(audible * .45, audible * .8)
    first = np.flatnonzero(valid)
    start = float(np.median(f0[first[:5]])) if first.size else None
    start_time_ms = _round(times[first[0]] * 1000, 1) if first.size else None
    checkpoints = {str(t): _round(median_window(max(0, t - .02), t + .02)) for t in (.05, .1, .2, .5, 1.0) if t <= duration}
    note = cents = None
    if steady:
        midi = 69 + 12 * math.log2(steady / 440)
        rounded = round(midi)
        note = librosa.midi_to_note(rounded)
        cents = _round((midi - rounded) * 100, 1)
    drop = None
    if steady and start:
        delta = start - steady
        if delta > max(3, start * .025):
            reached = np.flatnonzero(valid & (f0 <= start - .9 * delta))
            if reached.size:
                drop = {"start_hz": _round(start), "end_hz": _round(steady),
                        "time_to_90pct_ms": _round(times[reached[0]] * 1000, 1)}
    return {"f0_hz": [_round(v, 2) for v in f0], "times_s": [_round(t, 4) for t in times], "voiced": [bool(v) for v in voiced],
            "f0_start": _round(start), "f0_start_time_ms": start_time_ms, "f0_at": checkpoints, "f0_steady": _round(steady),
            "note": note, "cents": cents, "pitch_drop": drop}


def envelope_features(y: np.ndarray, sr: int, f0: float | None = None) -> dict[str, Any]:
    """Return peak-envelope timings in milliseconds and a one-shot estimate.

    ``attack_ms`` runs from 10% of the peak to -3 dB below it: a 90% threshold is unstable on soft, flat tops.

    The envelope is the Hilbert magnitude smoothed over one period of ``f0`` (at least 10 ms) and sampled
    every 5 ms. Short windows ripple on bass notes (one period of 33 Hz is 30 ms) and on harmonic-rich
    waves, which corrupts attack and decay times.
    """
    hop = max(1, int(sr * .005))
    env = uniform_filter1d(np.abs(hilbert(y)), size=max(1, int(sr * max(.01, 1 / f0 if f0 else 0))))[::hop]
    times = np.arange(len(env)) * hop / sr
    peak = max(float(env.max()), 1e-12)
    norm = env / peak
    peak_i = int(np.argmax(env))

    def first_after(level: float, start: int = 0) -> float | None:
        ids = np.flatnonzero(norm[start:] <= level)
        return float(times[start + ids[0]] * 1000) if ids.size else None

    i10 = np.flatnonzero(norm >= .1)
    i90 = np.flatnonzero(norm >= 10 ** (-3 / 20))
    attack = (times[i90[0]] - times[i10[0]]) * 1000 if i10.size and i90.size else None
    after = 20 * np.log10(np.maximum(norm[peak_i:], 1e-12))
    below = np.flatnonzero(after < -40)  # the noise floor below -40 dB wobbles, so judge monotonicity above it
    audible = after[: below[0]] if below.size else after
    one_shot = bool(after[-1] < -20 and np.all(audible <= np.minimum.accumulate(audible) + 3))
    return {"env_times_s": [_round(t, 4) for t in times], "env_dbfs": [_round(_db(v), 2) for v in env],
            "attack_ms": _round(attack, 1), "peak_time_ms": _round(times[peak_i] * 1000, 1),
            "decay_to_minus6_ms": _round(first_after(10 ** (-6 / 20), peak_i), 1),
            "decay_to_minus20_ms": _round(first_after(.1, peak_i), 1), "decay_to_minus40_ms": _round(first_after(.01, peak_i), 1),
            "tail_to_minus60_ms": _round(first_after(.001, peak_i), 1),
            "sustain_level_db": _round(_db(float(np.median(env[len(env) // 2:])) / peak), 1), "is_one_shot": one_shot}


def spectrum_features(y: np.ndarray, sr: int, f0: float | None) -> dict[str, Any]:
    """Return band percentages, spectral descriptors and harmonic levels in dB."""
    n_fft = min(4096, 2 ** int(math.floor(math.log2(max(256, len(y)))))); hop = max(64, n_fft // 4)
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)); freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # weight by frame energy: the centroid of near-silent tail frames is noise and would dominate a plain mean
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    frame_energy = np.sum(mag**2, axis=0)
    power = np.mean(mag**2, axis=1); total = max(float(power.sum()), 1e-18)
    bands = {"sub": (0,60), "bass":(60,250), "low_mid":(250,2000), "high_mid":(2000,6000), "high":(6000,sr/2)}
    result = {"band_energy_pct": {k: _round(100 * power[(freqs >= a)&(freqs < b)].sum()/total, 1) for k,(a,b) in bands.items()},
              "centroid_hz": _round(float(np.average(centroid, weights=frame_energy + 1e-18))),
              "rolloff_hz": _round(float(np.mean(librosa.feature.spectral_rolloff(S=mag, sr=sr)))),
              "flatness": _round(float(np.mean(librosa.feature.spectral_flatness(S=mag))))}
    harmonics: list[float | None] = []
    if f0 and f0 > 0:
        fundamental = np.interp(f0, freqs, np.mean(mag, axis=1))
        for h in range(1, 17):
            amp = np.interp(f0*h, freqs, np.mean(mag, axis=1)) if f0*h < sr/2 else 0
            harmonics.append(_round(20*np.log10(max(amp,1e-12)/max(fundamental,1e-12)),1))
        even = sum(10**((v or -120)/20) for v in harmonics[1::2]); odd = sum(10**((v or -120)/20) for v in harmonics[0::2])
        result["harmonics_db_relative"] = harmonics; result["even_odd_ratio"] = _round(even/max(odd,1e-12),3)
    else: result["harmonics_db_relative"] = []; result["even_odd_ratio"] = None
    return result


def extract_features(y: np.ndarray, channels: np.ndarray, sr: int, *, fmin: float = 20, fmax: float = 2000) -> dict[str, Any]:
    """Extract all deterministic report features from float32 mono audio."""
    pitch = pitch_features(y, sr, fmin, fmax)
    envelope = envelope_features(y, sr, pitch["f0_steady"])
    spectrum = spectrum_features(y, sr, pitch["f0_steady"])
    crest = float(np.max(np.abs(y)) / max(np.sqrt(np.mean(y**2)), 1e-12))
    return {"meta": meta_features(y, channels, sr), "pitch": pitch, "envelope": envelope, "spectrum": spectrum,
            "distortion": {"crest_factor_db": _round(_db(crest), 2), "clipping": bool(np.max(np.abs(y)) >= .999), "thd_estimate": None},
            "modulation": {"pitch": None, "amplitude": None}, "onset_rhythm": {"onsets_s": [], "tempo_bpm": None}}
