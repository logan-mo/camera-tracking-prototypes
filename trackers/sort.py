"""SORT: constant-velocity Kalman + IoU association, from scratch.

No scipy, no filterpy, no lap.  The Kalman filter is ~60 lines of NumPy and the
assignment step is a Jonker-Volgenant shortest-augmenting-path solver -- the same
O(n^3) algorithm ``scipy.optimize.linear_sum_assignment`` uses.

Why JV and not greedy
---------------------
For the single-target clips here greedy is *provably* identical: with at most
one detection and one track the problem is 1x1 and there is exactly one feasible
assignment.  JV is shipped as the default anyway because it is 45 lines once;
because the occlusion clips and any future YOLO backend do produce
multi-candidate frames where greedy can lock a globally worse pairing that is
very hard to tell apart from a tracker bug; and because having the provably
optimal solver removes "is my baseline itself wrong?" from the debugging
surface.  ``--assign greedy`` is exposed for A/B.

Note on the results table
-------------------------
SORT's box comes from the detector every frame, so it needs no scale handling
and will score near-zero scale error.  That is **not** SORT beating DSST at
scale estimation -- it is measuring the detector.  The ``Type`` column exists to
keep that distinction visible.
"""

from __future__ import annotations

import numpy as np

from .detectors import Detector
from .metrics import iou_batch
from .types import BBox, Detection

__all__ = ["KalmanBoxTracker", "Sort", "SortTracker", "greedy_assignment", "linear_assignment"]


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------
def linear_assignment(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment via Jonker-Volgenant. Returns (K, 2) [row, col].

    Rectangular-safe.  Costs should be non-negative and finite; feed
    ``1 - iou`` rather than ``-iou`` so that holds by construction.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    transposed = cost.shape[0] > cost.shape[1]
    a = cost.T.copy() if transposed else cost.copy()
    n, m = a.shape

    inf = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row assigned to column j (1-indexed)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            # Vectorised scan over unused columns -- the only inner loop.
            cols = np.flatnonzero(~used[1:]) + 1
            if cols.size == 0:
                break
            cur = a[i0 - 1, cols - 1] - u[i0] - v[cols]
            better = cur < minv[cols]
            if better.any():
                minv[cols[better]] = cur[better]
                way[cols[better]] = j0
            k = int(np.argmin(minv[cols]))
            delta = float(minv[cols[k]])
            j1 = int(cols[k])

            used_cols = np.flatnonzero(used)
            u[p[used_cols]] += delta
            v[used_cols] -= delta
            minv[cols] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augment along the alternating path.
        while j0 != 0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    out = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return out[:, ::-1].copy() if transposed else out


def greedy_assignment(cost: np.ndarray) -> np.ndarray:
    """Greedy nearest-cost matching. Exposed for A/B against :func:`linear_assignment`."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        r, c = int(r), int(c)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------
# Kalman filter
# --------------------------------------------------------------------------
def bbox_to_z(bbox: BBox) -> np.ndarray:
    """(x, y, w, h) -> (u, v, s, r) = centre, area, aspect."""
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0, w * h, w / h], dtype=np.float64).reshape(4, 1)


def z_to_bbox(z: np.ndarray) -> BBox:
    """(u, v, s, r) -> (x, y, w, h), guarding the classic SORT NaN."""
    u, v, s, r = (float(z[i]) for i in range(4))
    # s is an *area* and r an aspect; a coasting track whose area velocity has
    # eaten the area drives s <= 0, and sqrt(s*r) then yields NaN boxes that
    # propagate silently through every metric.
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = s / w if w > 0 else 1e-3
    return (u - w / 2.0, v - h / 2.0, w, h)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter on [u, v, s, r, u', v', s']."""

    count: int = 0

    @classmethod
    def reset_count(cls) -> None:
        """Required for run-to-run determinism of track ids."""
        cls.count = 0

    def __init__(self, bbox: BBox, score: float = 1.0) -> None:
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Bewley's reference values, stated explicitly so nothing is guessed.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])

        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.score = float(score)
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self) -> BBox:
        # Zero the area velocity when it would drive the area negative.
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.age += 1
        self.time_since_update += 1
        return self.state

    def update(self, det: Detection) -> None:
        z = bbox_to_z(det.bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # solve rather than an explicit 4x4 inverse
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(det.score)

    @property
    def state(self) -> BBox:
        return z_to_bbox(self.x[:4, 0])

    def confirmed(self, min_hits: int) -> bool:
        return self.hit_streak >= min_hits


class Sort:
    """Multi-object tracker: predict, associate by IoU, update, age out."""

    def __init__(
        self,
        max_age: int = 8,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        min_score: float = 0.0,
        assign: str = "jv",
    ) -> None:
        # Reference SORT uses max_age=1.  8 frames (~0.27 s at 30 fps) lets a
        # track coast behind an occluder instead of being reborn with a new id
        # on the far side; it is the single knob that decides whether SORT
        # "handles" the occlusion clips, so it is exposed rather than buried.
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.min_score = float(min_score)
        self.assign = assign
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0
        KalmanBoxTracker.reset_count()

    def _solve(self, cost: np.ndarray) -> np.ndarray:
        return greedy_assignment(cost) if self.assign == "greedy" else linear_assignment(cost)

    def update(self, detections: list[Detection]) -> np.ndarray:
        """Advance one frame. Returns (K, 5) rows of [x, y, w, h, id]."""
        self.frame_count += 1
        dets = [d for d in detections if d.score >= self.min_score]

        predicted: list[BBox] = []
        alive: list[KalmanBoxTracker] = []
        for track in self.tracks:
            bbox = track.predict()
            if np.all(np.isfinite(bbox)):
                predicted.append(bbox)
                alive.append(track)
        self.tracks = alive

        matched: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(dets)))
        if dets and predicted:
            det_xyxy = np.array([d.to_xyxy() for d in dets])
            trk_xyxy = np.array(
                [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in predicted]
            )
            iou_matrix = iou_batch(det_xyxy, trk_xyxy)
            pairs = self._solve(1.0 - iou_matrix)
            unmatched_dets = []
            claimed = set()
            for d_i, t_i in pairs:
                if iou_matrix[d_i, t_i] >= self.iou_threshold:
                    matched.append((int(d_i), int(t_i)))
                    claimed.add(int(d_i))
            unmatched_dets = [i for i in range(len(dets)) if i not in claimed]

        for d_i, t_i in matched:
            self.tracks[t_i].update(dets[d_i])
        for d_i in unmatched_dets:
            self.tracks.append(KalmanBoxTracker(dets[d_i].bbox, dets[d_i].score))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        rows = [
            [*t.state, float(t.id)]
            for t in self.tracks
            if t.time_since_update < 1
            and (t.confirmed(self.min_hits) or self.frame_count <= self.min_hits)
        ]
        return np.array(rows, dtype=np.float64).reshape(-1, 5)


class SortTracker:
    """Single-target adapter so SORT fits the bench's tracker interface."""

    name = "SORT"
    type = "detection"
    supports_scale = True

    def __init__(self, detector: Detector, **sort_kwargs) -> None:
        self.detector = detector
        self.sort = Sort(**sort_kwargs)
        self._id: int | None = None
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)
        self._confidence = 0.0
        self.diagnostics: dict[str, object] = {}

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def psr(self) -> float:
        return float("nan")

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self.sort.reset()
        self._bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        tracks = self.sort.update(self.detector.detect(frame))
        self._id = None
        if len(tracks):
            det_xyxy = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])
            trk_xyxy = np.column_stack(
                [tracks[:, 0], tracks[:, 1], tracks[:, 0] + tracks[:, 2], tracks[:, 1] + tracks[:, 3]]
            )
            best = int(np.argmax(iou_batch(det_xyxy, trk_xyxy)[0]))
            self._id = int(tracks[best, 4])
            self._bbox = tuple(float(v) for v in tracks[best, :4])  # type: ignore[assignment]
        self._confidence = 1.0
        self.diagnostics = {"n_tracks": len(tracks), "locked_id": self._id}

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        detections = self.detector.detect(frame)
        tracks = self.sort.update(detections)
        self.diagnostics = {"n_tracks": len(tracks), "n_dets": len(detections)}

        if len(tracks) == 0:
            self._confidence = 0.0
            return False, self._bbox

        ids = tracks[:, 4].astype(np.int64)
        if self._id is not None and self._id in ids:
            row = tracks[int(np.flatnonzero(ids == self._id)[0])]
            self._bbox = tuple(float(v) for v in row[:4])  # type: ignore[assignment]
            self._confidence = 1.0
            return True, self._bbox

        # The locked track died.  Re-lock to the nearest surviving one, but
        # report ok=False for this frame so frames-to-failure stays honest.
        cx, cy = self._bbox[0] + self._bbox[2] / 2, self._bbox[1] + self._bbox[3] / 2
        centres = np.column_stack([tracks[:, 0] + tracks[:, 2] / 2, tracks[:, 1] + tracks[:, 3] / 2])
        nearest = int(np.argmin(np.linalg.norm(centres - np.array([cx, cy]), axis=1)))
        self._id = int(tracks[nearest, 4])
        self._bbox = tuple(float(v) for v in tracks[nearest, :4])  # type: ignore[assignment]
        self._confidence = 0.0
        return False, self._bbox
