"""Distance and advice on synthetic renders."""

from pathlib import Path

import numpy as np
import soundfile as sf

from soundkb.analysis.compare import compare_audio_files

SR = 44100


def _tone(path: Path, freq=55.0, decay_tau=0.6, harmonics=(), duration=2.0):
    t = np.arange(int(SR * duration)) / SR
    y = np.sin(2 * np.pi * freq * t)
    for n, gain in harmonics:
        y += gain * np.sin(2 * np.pi * freq * n * t)
    y = 0.5 * y / max(np.max(np.abs(y)), 1) * np.exp(-t / decay_tau) * np.minimum(t / 0.01, 1)
    sf.write(path, y.astype(np.float32), SR)
    return path


def test_identical_is_zero(tmp_path):
    a = _tone(tmp_path / "a.wav")
    result = compare_audio_files(a, a)
    assert result["distance"] < 0.05
    assert result["advice"] == []


def test_shorter_decay_and_detuned_is_worse_and_explained(tmp_path):
    ref = _tone(tmp_path / "ref.wav", freq=55.0, decay_tau=0.8)
    near = _tone(tmp_path / "near.wav", freq=55.0, decay_tau=0.7)
    far = _tone(tmp_path / "far.wav", freq=61.0, decay_tau=0.2)
    d_near = compare_audio_files(ref, near)["distance"]
    far_result = compare_audio_files(ref, far)
    assert far_result["distance"] > d_near + 0.3
    text = " ".join(far_result["advice"]).lower()
    assert "cents" in text and "decay" in text


def test_missing_harmonics_hint(tmp_path):
    ref = _tone(tmp_path / "ref.wav", harmonics=[(3, 0.4), (5, 0.3)])
    cand = _tone(tmp_path / "cand.wav")
    assert any("harmonics" in line.lower() for line in compare_audio_files(ref, cand)["advice"])
