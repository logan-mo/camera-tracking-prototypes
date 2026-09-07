"""Shared geometry types for the tracking bench.

This module is the SOLE OWNER of the bbox convention and of ``Detection``.
Every other module imports from here; nothing redefines them.

Convention, stated once:

    BBox = (x, y, w, h)   top-left corner + size, floats.

That matches ``cv.boundingRect`` and ``cv.selectROI``.  The corner/size form is
the external convention everywhere -- in signatures, CSVs, sidecars and the
overlay.  Corner/corner (xyxy) appears only inside vectorised IoU, and
centre/size only inside a tracker's own state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BBox = tuple[float, float, float, float]

__all__ = [
    "BBox",
    "Detection",
    "bbox_area",
    "bbox_centre",
    "bbox_from_centre",
    "clip_bbox",
    "is_valid_bbox",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
]


def xywh_to_xyxy(b: BBox) -> tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def xyxy_to_xywh(b: tuple[float, float, float, float]) -> BBox:
    x0, y0, x1, y1 = b
    return (x0, y0, x1 - x0, y1 - y0)


def bbox_centre(b: BBox) -> tuple[float, float]:
    """Centre of ``b``.

    Uses the continuous convention ``cx = x + w/2`` rather than ``x + (w-1)/2``.
    The ``-1`` variant accumulates a systematic bias once DSST starts
    multiplying sizes by a float scale factor, so it is avoided everywhere.
    """
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def bbox_from_centre(cx: float, cy: float, w: float, h: float) -> BBox:
    return (cx - w / 2.0, cy - h / 2.0, w, h)


def bbox_area(b: BBox) -> float:
    _, _, w, h = b
    return max(0.0, float(w)) * max(0.0, float(h))


def is_valid_bbox(b: BBox | None) -> bool:
    """True when ``b`` is a finite box with positive extent."""
    if b is None:
        return False
    if len(b) != 4:
        return False
    x, y, w, h = b
    if not all(math.isfinite(float(v)) for v in (x, y, w, h)):
        return False
    return w > 0.0 and h > 0.0


def clip_bbox(b: BBox, width: int, height: int) -> BBox:
    """Clip ``b`` to a ``width`` x ``height`` frame, preserving corner/size form."""
    x0, y0, x1, y1 = xywh_to_xyxy(b)
    x0 = min(max(x0, 0.0), float(width))
    y0 = min(max(y0, 0.0), float(height))
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    return xyxy_to_xywh((x0, y0, max(x0, x1), max(y0, y1)))


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector output.

    Lives here rather than in ``detectors`` so that ``sort`` can import it
    without importing any detector implementation.  That is the seam which lets
    a YOLO backend drop in later without touching SORT.
    """

    bbox: BBox
    score: float = 1.0
    class_id: int = 0

    def to_xyxy(self) -> np.ndarray:
        return np.asarray(xywh_to_xyxy(self.bbox), dtype=np.float64)

    @property
    def centre(self) -> tuple[float, float]:
        return bbox_centre(self.bbox)
