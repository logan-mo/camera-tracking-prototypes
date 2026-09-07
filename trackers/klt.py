"""Sparse optical flow (Shi-Tomasi + pyramidal Lucas-Kanade) as a tracker.

This is the productionised form of ``sparse-optical-flow.py``.  It is the
cheapest baseline in the bench and, per spec, has **no scale handling and no
occlusion handling at all**.

Four deliberate changes from the tutorial, each for a reason:

1. **Features are seeded inside the init box only**, via a mask.  The tutorial
   detects over the whole frame, which on a uniform grey-15 background would put
   corners on the arrow overlay in the source clip and on codec ringing rather
   than on the target.
2. **``qualityLevel`` 0.3 -> 0.01 and ``minDistance`` 7 -> 5.**  A smooth
   antialiased white disc on flat grey has a weak Shi-Tomasi response everywhere
   except its rim; at 0.3 only a handful of corners survive on a 35 px circle,
   which is below ``min_points``, and the tracker dies at init.
3. **Forward-backward (Kalal) consistency check.**  Track forward, then back,
   and keep only points that return to where they started.  This is what stops
   the point cloud silently sliding onto the background.
4. **Robust outlier rejection** on displacement (median absolute deviation).

Box derivation
--------------
``bbox_mode="rigid"`` is the default: the box size is frozen at its init value
and only the centre moves.  A point-cloud hull would be tempting, but corners
live strictly *inside* the object and every point that dies shrinks the hull
further, so the hull reports a steadily shrinking box.  That artifact would show
up as *fake scale adaptation* in ``ScaleErr`` -- the exact column this bench
turns on -- so it is not the default.  ``"bbox"`` (hull) is available precisely
so the benchmark can demonstrate why it is worse, and ``"similarity"`` estimates
scale from the median pairwise-distance ratio (the MedianFlow estimate), which
is the honest answer to "what is the cheapest way to add scale?".

Failure mode
------------
When points die out, ``update`` returns ``(False, last_bbox)`` and freezes
forever -- there is no re-detection, by design.  ``reseed=True`` re-runs corner
detection inside the last box, but it only recovers if the box is still on the
target; otherwise it locks onto background permanently, which is worse than
failing loudly.  Default off.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .types import BBox, bbox_centre, bbox_from_centre, clip_bbox

__all__ = ["KLTTracker"]


class KLTTracker:
    name = "KLT"
    type = "flow"
    supports_scale = False

    def __init__(
        self,
        max_corners: int = 100,
        quality_level: float = 0.01,
        min_distance: float = 5.0,
        block_size: int = 7,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        bbox_mode: str = "rigid",
        min_points: int = 4,
        fb_threshold: float = 1.0,
        err_threshold: float = 20.0,
        mad_scale: float = 2.5,
        reseed: bool = False,
    ) -> None:
        if bbox_mode not in {"rigid", "bbox", "similarity"}:
            raise ValueError(f"unknown bbox_mode {bbox_mode!r}")
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.bbox_mode = bbox_mode
        self.min_points = int(min_points)
        self.fb_threshold = float(fb_threshold)
        self.err_threshold = float(err_threshold)
        self.mad_scale = float(mad_scale)
        self.reseed = bool(reseed)

        if bbox_mode == "similarity":
            self.name = "KLT-S"
            self.supports_scale = True

        self._prev: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)
        self._size = (1.0, 1.0)
        self._n_init = 0
        self._dead = False
        self.diagnostics: dict[str, object] = {}

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def _detect(self, gray: np.ndarray, bbox: BBox) -> np.ndarray | None:
        h, w = gray.shape[:2]
        x, y, bw, bh = clip_bbox(bbox, w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + bw)), int(round(y + bh))
        if x1 <= x0 or y1 <= y0:
            return None
        mask[y0:y1, x0:x1] = 255
        return cv.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)

    @property
    def confidence(self) -> float:
        if self._n_init == 0 or self._points is None:
            return 0.0
        return float(len(self._points)) / float(self._n_init)

    @property
    def psr(self) -> float:
        return float("nan")

    # -- interface ----------------------------------------------------------
    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        gray = self._gray(frame)
        self._bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        self._size = (float(bbox[2]), float(bbox[3]))
        self._prev = gray
        self._dead = False

        points = self._detect(gray, bbox)
        if points is None or len(points) < self.min_points:
            # Degenerate init must not raise -- the bench needs a row for it.
            self._points = None
            self._n_init = 0
            self._dead = True
            self.diagnostics = {"n_points": 0, "reason": "insufficient features at init"}
            return

        self._points = points.reshape(-1, 1, 2).astype(np.float32)
        self._n_init = len(self._points)
        self._centre = bbox_centre(bbox)
        self.diagnostics = {"n_points": self._n_init}

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        gray = self._gray(frame)
        if self._dead or self._points is None or self._prev is None:
            return False, self._bbox

        prev_pts = self._points
        next_pts, status, err = cv.calcOpticalFlowPyrLK(
            self._prev, gray, prev_pts, None, **self.lk_params
        )
        if next_pts is None:
            self._dead = True
            return False, self._bbox

        # Forward-backward check: track the result back and keep only points
        # that return to their origin.
        back_pts, _, _ = cv.calcOpticalFlowPyrLK(
            gray, self._prev, next_pts, None, **self.lk_params
        )
        keep = status.reshape(-1) == 1
        if err is not None:
            keep &= err.reshape(-1) < self.err_threshold
        if back_pts is not None:
            fb = np.linalg.norm(
                prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            keep &= fb < self.fb_threshold

        good_prev = prev_pts.reshape(-1, 2)[keep]
        good_next = next_pts.reshape(-1, 2)[keep]

        if len(good_next) >= 3:
            # Robust outlier rejection on displacement.
            disp = good_next - good_prev
            median = np.median(disp, axis=0)
            dist = np.linalg.norm(disp - median, axis=1)
            mad = np.median(np.abs(dist - np.median(dist)))
            if mad > 1e-6:
                inliers = dist <= np.median(dist) + self.mad_scale * mad * 1.4826
                if inliers.sum() >= self.min_points:
                    good_prev, good_next = good_prev[inliers], good_next[inliers]

        if len(good_next) < self.min_points:
            if self.reseed:
                points = self._detect(gray, self._bbox)
                if points is not None and len(points) >= self.min_points:
                    self._points = points.reshape(-1, 1, 2).astype(np.float32)
                    self._prev = gray
                    # Centre is carried over unchanged: the reseeded cloud is a
                    # fresh set with no pairing to the old one, so there is no
                    # displacement to integrate this frame.
                    self._centre = bbox_centre(self._bbox)
                    self.diagnostics = {"n_points": len(self._points), "reseeded": True}
                    return False, self._bbox
            self._dead = True
            self.diagnostics = {"n_points": int(len(good_next)), "reason": "points died out"}
            return False, self._bbox

        bbox = self._derive_bbox(good_prev, good_next, gray.shape[:2])
        self._bbox = bbox
        self._points = good_next.reshape(-1, 1, 2).astype(np.float32)
        self._prev = gray
        self.diagnostics = {"n_points": int(len(good_next))}
        return True, bbox

    def _derive_bbox(self, prev: np.ndarray, cur: np.ndarray,
                     shape: tuple[int, int]) -> BBox:
        h, w = shape
        if self.bbox_mode == "bbox":
            x0, y0 = cur.min(axis=0)
            x1, y1 = cur.max(axis=0)
            return clip_bbox((float(x0), float(y0), float(x1 - x0), float(y1 - y0)), w, h)

        # Integrate the median *displacement* of paired surviving points rather
        # than tracking the cloud's absolute position.  Point attrition is
        # asymmetric -- the FB and MAD filters cull one side of the rim more than
        # the other -- so an absolute centroid jumps every time a point dies,
        # which showed up as a constant ~15 px bias locked in over the first
        # dozen frames.  A paired displacement only ever uses points present in
        # both frames, so culling cannot shift it.
        # Median rather than mean: robust to a single surviving outlier.
        delta = np.median(cur - prev, axis=0)
        cx = self._centre[0] + float(delta[0])
        cy = self._centre[1] + float(delta[1])
        self._centre = (cx, cy)

        if self.bbox_mode == "similarity" and len(cur) >= 2:
            i, j = np.triu_indices(len(cur), k=1)
            d_prev = np.linalg.norm(prev[i] - prev[j], axis=1)
            d_cur = np.linalg.norm(cur[i] - cur[j], axis=1)
            valid = d_prev > 1e-3
            if valid.any():
                ratio = float(np.median(d_cur[valid] / d_prev[valid]))
                ratio = float(np.clip(ratio, 0.9, 1.1))  # per-frame scale is small
                self._size = (self._size[0] * ratio, self._size[1] * ratio)

        return clip_bbox(bbox_from_centre(cx, cy, *self._size), w, h)
