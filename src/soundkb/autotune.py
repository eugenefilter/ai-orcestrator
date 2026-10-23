"""Coordinate-descent patch tuning against a reference sound.

Each evaluation is one real render, so the budget is small: every parameter is probed at value +- step and the
best of the three is kept; steps halve after each pass; two passes that improve by less than ``min_gain`` (relative) stop the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ParamSpec:
    index: int
    lo: float
    hi: float
    value: float
    step: float
    name: str = ""


def optimize(params: list[ParamSpec], evaluate: Callable[[dict[int, float]], float], max_renders: int = 24,
             min_gain: float = .01) -> dict:
    """Minimise ``evaluate`` (distance) over the parameters. Returns best values, best distance and the full history."""
    values = {p.index: p.value for p in params}
    best = evaluate(values)
    renders = 1
    history = [{"values": dict(values), "distance": best}]
    stale = 0
    while renders < max_renders and stale < 2:
        pass_start = best
        for p in params:
            for candidate in (min(p.hi, values[p.index] + p.step), max(p.lo, values[p.index] - p.step)):
                if renders >= max_renders or candidate == values[p.index]:
                    continue
                trial = {**values, p.index: candidate}
                distance = evaluate(trial)
                renders += 1
                history.append({"values": dict(trial), "distance": distance})
                if distance < best:
                    best, values = distance, trial
        stale = stale + 1 if pass_start - best < min_gain * pass_start else 0  # two passes without real gain end the search
        for p in params:
            p.step /= 2
    return {"values": values, "distance": best, "renders": renders, "history": history}
