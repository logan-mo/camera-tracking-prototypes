"""Camera tracking prototypes: six trackers behind one interface, plus a bench.

OpenCV 5.0 removed every classical tracker (``cv2.legacy``, ``TrackerKCF``,
``TrackerCSRT``, ``TrackerMOSSE`` are all gone), so MOSSE, KCF, DSST and CSRT
here are from-scratch NumPy implementations rather than OpenCV wrappers.  DSST
was never in mainline OpenCV at all.

Nothing cv2-heavy is imported at package import time -- tracker modules are
resolved lazily by :mod:`trackers.registry`, which is what lets the benchmark
matrix skip a tracker that does not exist yet instead of crashing.
"""

from .types import BBox, Detection

__all__ = ["BBox", "Detection", "available", "build"]


def build(name: str, **kwargs):
    """Instantiate a tracker by name (lazy import)."""
    from .registry import build as _build

    return _build(name, **kwargs)


def available() -> list[str]:
    """Tracker names that can be imported right now."""
    from .registry import available as _available

    return _available()
