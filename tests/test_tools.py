"""Calibration solver, autotune and recorder helpers (no Ableton or audio device needed)."""

import numpy as np
import pytest

from soundkb.autotune import ParamSpec, optimize
from soundkb.calibrate import solve
from soundkb.record import parse_audio_devices, trim_leading_silence

LISTING = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] Serato Virtual Audio
[AVFoundation indev @ 0x1] [1] BlackHole 2ch
"""


def test_parse_audio_devices_ignores_video():
    assert parse_audio_devices(LISTING) == {"Serato Virtual Audio": 0, "BlackHole 2ch": 1}


def test_trim_leading_silence():
    samples = np.zeros((1000, 2), dtype=np.float32)
    samples[300:, :] = .5
    trimmed, removed = trim_leading_silence(samples)
    assert removed == 300 and len(trimmed) == 700


def _calibration():
    # attack time grows exponentially with the parameter: 10 ms at 0.2, 100 ms at 0.5, 1000 ms at 0.8
    return {"min": 0.0, "max": 1.0, "points": [{"value": v, "metrics": {"attack_ms": t}} for v, t in ((.2, 10), (.5, 100), (.8, 1000))]}


def test_solve_interpolates_in_log_space():
    result = solve(_calibration(), "attack_ms", 31.6)
    assert result["value"] == pytest.approx(.35, abs=.01)
    assert not result["extrapolated"]


def test_solve_extrapolates_and_clamps():
    result = solve(_calibration(), "attack_ms", 100000)
    assert result["extrapolated"] and result["value"] <= 1.0
    with pytest.raises(ValueError, match="at least two"):
        solve({"min": 0, "max": 1, "points": [{"value": .1, "metrics": {"attack_ms": None}}]}, "attack_ms", 5)


def test_optimize_finds_minimum_within_budget():
    target = {0: .62, 1: .31}
    evaluate = lambda v: sum((v[i] - t) ** 2 for i, t in target.items())
    result = optimize([ParamSpec(0, 0, 1, .2, .2), ParamSpec(1, 0, 1, .8, .2)], evaluate, max_renders=40)
    assert result["distance"] < .01
    assert result["renders"] <= 40
