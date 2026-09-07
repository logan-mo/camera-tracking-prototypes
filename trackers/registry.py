"""Tracker registry with lazy imports.

Imports happen at ``build()`` time, not at module import.  That is what lets the
comparison matrix print ``SKIP (not implemented)`` for a tracker that does not
exist yet instead of crashing the whole run -- and it is what allowed the entire
harness to be built and validated with only the oracles present.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TrackerSpec", "available", "build", "needs_groundtruth", "register", "tracker_type"]


@dataclass(frozen=True)
class TrackerSpec:
    module: str
    attr: str
    type: str = "appearance"
    kwargs: dict[str, Any] = field(default_factory=dict)
    needs_gt: bool = False
    needs_detector: bool = False


TRACKER_SPECS: dict[str, TrackerSpec] = {}


def register(
    name: str,
    module: str,
    attr: str,
    *,
    type: str = "appearance",
    needs_gt: bool = False,
    needs_detector: bool = False,
    **kwargs: Any,
) -> None:
    TRACKER_SPECS[name.lower()] = TrackerSpec(
        module=module, attr=attr, type=type, kwargs=kwargs,
        needs_gt=needs_gt, needs_detector=needs_detector,
    )


# Appearance-based correlation filters (from scratch -- OpenCV 5.0 removed them all).
register("mosse", ".mosse", "MOSSE", type="appearance")
register("kcf", ".kcf", "KCF", type="appearance")
register("dsst", ".dsst", "DSST", type="appearance")
register("csrt", ".csrt", "CSRT", type="appearance")
# Sparse optical flow.
register("klt", ".klt", "KLTTracker", type="flow")
# Detection-based.  Its box follows the detector, so it is a different kind of
# thing from the appearance trackers -- hence the distinct type, which the
# results table surfaces.
register("sort", ".sort", "SortTracker", type="detection", needs_detector=True)
# References that bound the table.
register("oracle", ".oracles", "OracleTracker", type="reference", needs_gt=True)
register("fixedbox", ".oracles", "FixedBoxOracle", type="reference", needs_gt=True)
register("static", ".oracles", "StaticOracle", type="reference", needs_gt=True)

DEFAULT_ORDER = [
    "oracle", "sort", "csrt", "dsst", "fixedbox", "kcf", "mosse", "klt", "static",
]


def _resolve(spec: TrackerSpec):
    module = importlib.import_module(spec.module, package=__package__)
    return getattr(module, spec.attr)


def build(name: str, **overrides: Any):
    """Instantiate a tracker by name. Raises ImportError if not yet implemented."""
    key = name.lower()
    if key not in TRACKER_SPECS:
        raise KeyError(f"unknown tracker {name!r}; known: {', '.join(sorted(TRACKER_SPECS))}")
    spec = TRACKER_SPECS[key]
    cls = _resolve(spec)
    return cls(**{**spec.kwargs, **overrides})


def available() -> list[str]:
    """Names that can actually be imported right now, in display order."""
    out = []
    for name in DEFAULT_ORDER + sorted(set(TRACKER_SPECS) - set(DEFAULT_ORDER)):
        if name in out:
            continue
        try:
            _resolve(TRACKER_SPECS[name])
        except (ImportError, AttributeError):
            continue
        out.append(name)
    return out


def tracker_type(name: str) -> str:
    spec = TRACKER_SPECS.get(name.lower())
    return spec.type if spec else "appearance"


def needs_groundtruth(name: str) -> bool:
    spec = TRACKER_SPECS.get(name.lower())
    return bool(spec and spec.needs_gt)


def needs_detector(name: str) -> bool:
    spec = TRACKER_SPECS.get(name.lower())
    return bool(spec and spec.needs_detector)
