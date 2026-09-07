"""Reference "trackers" that bound the results table.

These are not trackers.  They exist so that a surprising number in the results
table can be attributed to a tracker rather than to the bench, which is the
ambiguity that otherwise sinks this kind of comparison.

:class:`OracleTracker`
    Returns ground truth every frame.  **Must** score IoU 1.00, AUC 1.00,
    ScaleErr 0.00, Prec@20 1.00 on every clip.  Any deviation is a metrics bug,
    found while the search space is still tiny.

:class:`FixedBoxOracle`
    Perfect centre, frozen box size.  This is the analytic prediction made
    executable: on ``synth_scale.mp4`` (radius sweeping 12<->48 px about a
    mid-scale init) it must score **ScaleErr ~ 0.637 octaves**, **Prec@20 1.00**
    and **IoU ~ 0.55**, because ``mean|log2(r_init/r(t))| = ln(2)/ln(2) * 2/pi``.

    With both oracles in the table it is bounded at each end.  KCF and MOSSE
    landing near ``FixedBox`` on ScaleErr is then the *expected, correct* result
    rather than a suspected bug -- and if DSST fails to beat ``FixedBox`` on
    ScaleErr, its scale filter is broken, which is known without ever arguing
    about "tracker quality".

:class:`StaticOracle`
    Init box, never moves.  The null baseline: any real tracker must beat it.
"""

from __future__ import annotations

import numpy as np

from .groundtruth import GroundTruth
from .types import BBox, bbox_from_centre

__all__ = ["FixedBoxOracle", "OracleTracker", "StaticOracle"]


class _OracleBase:
    type = "reference"
    supports_scale = False

    def __init__(self, gt: GroundTruth) -> None:
        self.gt = gt
        self._index = 0
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)
        self._init_size = (1.0, 1.0)
        self.diagnostics: dict[str, object] = {}
        self.confidence = 1.0
        self.psr = float("nan")

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self._init_size = (float(bbox[2]), float(bbox[3]))
        self._index = 0

    def _advance(self) -> tuple[bool, BBox]:
        self._index += 1
        entry = self.gt[self._index]
        if entry.valid and entry.bbox is not None:
            return True, entry.bbox
        return False, self._bbox


class OracleTracker(_OracleBase):
    """Upper bound: returns ground truth verbatim."""

    name = "Oracle"
    supports_scale = True

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        ok, bbox = self._advance()
        self._bbox = bbox
        return ok, bbox


class FixedBoxOracle(_OracleBase):
    """Perfect translation, zero scale adaptation -- the fixed-box lower bound."""

    name = "FixedBox"
    supports_scale = False

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        ok, bbox = self._advance()
        if ok:
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            self._bbox = bbox_from_centre(cx, cy, *self._init_size)
        return ok, self._bbox


class StaticOracle(_OracleBase):
    """Null baseline: the init box, forever."""

    name = "Static"
    supports_scale = False

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        self._index += 1
        return True, self._bbox
