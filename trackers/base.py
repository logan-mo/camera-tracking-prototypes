"""The tracker interface every tracker in this bench satisfies.

Contract
--------
``init(frame_bgr, bbox)`` then ``update(frame_bgr) -> (ok, bbox)``.

``update`` takes the **BGR frame straight from ``cv.VideoCapture``**, not a
pre-converted grayscale image.  Each tracker converts internally and caches,
because CSRT needs colour for its histogram model even when a particular clip
happens to be monochrome.

``ok=False`` means "low confidence, this box may be stale".  A tracker still
returns its best available box; it never returns ``None``.  What to do about a
stale box is the caller's decision, and the runner records it so
frames-to-failure stays honest.

The shared update gate
----------------------
``_should_update`` lives here, in the base class, and is used identically by all
four correlation-filter trackers.  Without a *uniform* update policy, a result
like "CSRT survives occlusion" could simply mean "CSRT happened to get a luckier
update rule", and the comparison would be measuring policy rather than
algorithm.

Confidence is deliberately **relative**, not an absolute PSR threshold.  The
clips here have a background with exactly zero variance (measured: mean 15.00,
std 0.000), so PSR's sidelobe denominator is near-zero and Bolme's canonical
``PSR < 7 -> lost`` rule would never fire -- not even during full occlusion.
Each tracker reports raw PSR too, so the content-dependence stays visible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .types import BBox, bbox_centre, bbox_from_centre, is_valid_bbox

__all__ = ["BaseTracker", "TrackerConfig"]


@dataclass
class TrackerConfig:
    """Knobs shared by every correlation-filter tracker."""

    padding: float = 1.5
    lr: float = 0.075
    lam: float = 1e-4
    output_sigma_factor: float = 0.1
    # Update gating.
    conf_window: int = 30
    conf_ratio: float = 0.4
    psr_floor: float = 5.0
    max_skip: int = 30
    debug: bool = False
    extra: dict = field(default_factory=dict)


class BaseTracker(ABC):
    name: str = "base"
    type: str = "appearance"
    supports_scale: bool = False

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.diagnostics: dict[str, object] = {}
        self._centre: tuple[float, float] = (0.0, 0.0)
        self._size: tuple[float, float] = (0.0, 0.0)
        self._psr = 0.0
        self._confidence = 1.0
        self._psr_history: deque[float] = deque(maxlen=self.config.conf_window)
        self._skips = 0
        self._frame_index = 0
        self._lost = False

    # -- geometry -----------------------------------------------------------
    @property
    def bbox(self) -> BBox:
        return bbox_from_centre(*self._centre, *self._size)

    def _set_bbox(self, bbox: BBox) -> None:
        if not is_valid_bbox(bbox):
            raise ValueError(f"invalid bbox: {bbox!r}")
        self._centre = bbox_centre(bbox)
        self._size = (float(bbox[2]), float(bbox[3]))

    # -- diagnostics --------------------------------------------------------
    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def psr(self) -> float:
        return self._psr

    def _record_confidence(self, psr: float) -> float:
        """Turn a raw PSR into a relative confidence in [0, 1].

        Compared against a running median of recent PSR rather than an absolute
        constant, because absolute PSR is strongly content-dependent and is
        meaningless on a zero-variance background.  An absolute floor is kept as
        a backstop for genuinely degenerate responses.
        """
        self._psr = float(psr)
        if not np.isfinite(psr):
            self._confidence = 0.0
            return 0.0
        if self._psr_history:
            baseline = float(np.median(self._psr_history))
            conf = float(np.clip(psr / baseline, 0.0, 1.0)) if baseline > 1e-9 else 1.0
        else:
            conf = 1.0
        if psr < self.config.psr_floor:
            conf = min(conf, 0.0 if psr <= 0 else psr / max(self.config.psr_floor, 1e-9) * 0.4)
        self._psr_history.append(float(psr))
        self._confidence = conf
        return conf

    def _should_update(self, conf: float) -> bool:
        """Shared gate: skip the model update when confidence collapses.

        Identical for every correlation-filter tracker, on purpose -- see the
        module docstring.
        """
        if conf >= self.config.conf_ratio:
            self._skips = 0
            return True
        self._skips += 1
        if self._skips >= self.config.max_skip:
            self._lost = True
        return False

    @property
    def lost(self) -> bool:
        return self._lost

    # -- interface ----------------------------------------------------------
    @abstractmethod
    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise on ``frame`` with the target at ``bbox``."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        """Advance one frame. Returns ``(ok, bbox)``; bbox is never ``None``."""

    def reinit(self, frame: np.ndarray, bbox: BBox) -> None:
        """Re-initialise from scratch (used by redetection experiments)."""
        self.init(frame, bbox)

    def _reset_state(self, bbox: BBox) -> None:
        self._set_bbox(bbox)
        self.diagnostics = {}
        self._psr = 0.0
        self._confidence = 1.0
        self._psr_history.clear()
        self._skips = 0
        self._frame_index = 0
        self._lost = False
