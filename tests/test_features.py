"""Feature extractors on synthetic signals."""

import numpy as np
import pytest

from soundkb.analysis.features import envelope_features, extract_features, pitch_features

SR = 44100


def _808(duration=2.5, f_start=40.0, f_end=33.0, drop_tau=0.12, attack=0.05, decay_tau=0.6):
    t = np.arange(int(SR * duration)) / SR
    freq = f_end + (f_start - f_end) * np.exp(-t / drop_tau)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    amp = np.minimum(t / attack, 1) * np.exp(-np.maximum(t - attack, 0) / decay_tau)
    return (0.8 * amp * np.sin(phase)).astype(np.float32)


def test_pitch_drop_and_steady_note():
    pitch = pitch_features(_808(), SR)
    assert pitch["f0_steady"] == pytest.approx(33, abs=1.0)
    assert pitch["note"] == "C1"
    assert pitch["pitch_drop"] is not None
    assert pitch["pitch_drop"]["start_hz"] > pitch["pitch_drop"]["end_hz"] + 3


def test_decayed_tail_is_not_reported_as_pitch():
    pitch = pitch_features(_808(duration=5.0, decay_tau=0.3), SR)
    tail = [f for t, f in zip(pitch["times_s"], pitch["f0_hz"]) if t > 3.5]
    assert all(f is None for f in tail)


def test_envelope_attack_and_one_shot():
    env = envelope_features(_808(attack=0.05), SR)
    assert env["attack_ms"] == pytest.approx(30, abs=15)
    assert env["is_one_shot"] is True
    assert env["decay_to_minus20_ms"] > env["decay_to_minus6_ms"]


def test_sustained_tone_is_not_one_shot():
    t = np.arange(SR * 2) / SR
    assert envelope_features((0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32), SR)["is_one_shot"] is False


def test_silence_has_no_pitch():
    y = np.zeros(SR, dtype=np.float32)
    features = extract_features(y, y[:, None], SR)
    assert features["pitch"]["f0_steady"] is None
