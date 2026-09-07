"""Tracking metrics, timing and result formatting.

Two design decisions here carry the whole comparison:

**Scale is reported as two numbers, because one lies.**  ``median_scale_ratio``
gives direction and bias (0.71 reads as "the box is 1.4x too small");
``scale_err`` gives magnitude as ``mean |log2(ratio)|`` in octaves, which is
symmetric and non-cancelling.  A plain ``mean(scale_ratio)`` is actively
misleading on a sinusoidal scale clip, because over- and under-estimates cancel
to ~1.0 for a tracker that is wildly wrong in both directions.  ``scale_err`` is
the column that separates KCF/MOSSE from DSST/CSRT.

**IoU and centre error are reported separately, never merged.**  A disc is
centrally symmetric, so under scale change a fixed-box tracker's correlation
peak stays well-centred -- its centre error looks fine and only its *box* is
wrong.  Merging the two would hide exactly the failure this bench exists to
measure.

No pandas and no matplotlib are installed, so tables are fixed-width ASCII and
plots are drawn with characters.  Per-frame CSVs are written for plotting
elsewhere.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from .types import BBox, bbox_area, bbox_centre, xywh_to_xyxy

__all__ = [
    "FrameRecord",
    "RunStats",
    "Timer",
    "ascii_series_plot",
    "ascii_success_plot",
    "centre_error",
    "format_table",
    "iou",
    "iou_batch",
    "scale_ratio",
    "summarise",
    "write_frames_csv",
    "write_summary_csv",
]

SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=np.float64)
FAILURE_IOU = 0.3
FAILURE_PATIENCE = 3
REACQUIRE_IOU = 0.5

# Threshold comparisons are made with this slack.  IoU of two *identical* boxes
# is not bit-exactly 1.0: the corner form computes ix as (x+w)-x, which differs
# from w by an ULP for large x, so a perfect match lands just below 1.0 and the
# t=1.0 endpoint of the success curve rejects it.  Without the slack the Oracle
# -- whose whole job is to score exactly 1.000 and thereby validate the harness
# -- reports AUC 0.98, which reads as a metrics bug that isn't one.
THRESHOLD_EPS = 1e-9


def iou(a: BBox | None, b: BBox | None) -> float:
    if a is None or b is None:
        return float("nan")
    ax0, ay0, ax1, ay1 = xywh_to_xyxy(a)
    bx0, by0, bx1, by1 = xywh_to_xyxy(b)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    # Clamped because float error in the corner round-trip can push the ratio a
    # hair outside [0, 1] for near-identical boxes.
    return float(min(1.0, max(0.0, inter / union)))


def iou_batch(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays -> (N,M)."""
    a = np.asarray(a_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0.0, None) * np.clip(iy1 - iy0, 0.0, None)

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def centre_error(a: BBox | None, b: BBox | None) -> float:
    if a is None or b is None:
        return float("nan")
    ax, ay = bbox_centre(a)
    bx, by = bbox_centre(b)
    return float(math.hypot(ax - bx, ay - by))


def scale_ratio(pred: BBox | None, gt: BBox | None) -> float:
    """Linear scale factor ``sqrt(area_pred / area_gt)``.

    Linear rather than areal so the number reads directly as "the box is N times
    too wide", which is what a reader expects from a column called scale.
    """
    if pred is None or gt is None:
        return float("nan")
    ga = bbox_area(gt)
    pa = bbox_area(pred)
    if ga <= 0 or pa <= 0:
        return float("nan")
    return float(math.sqrt(pa / ga))


def normalised_centre_error(pred: BBox | None, gt: BBox | None) -> float:
    if pred is None or gt is None:
        return float("nan")
    ga = bbox_area(gt)
    if ga <= 0:
        return float("nan")
    return centre_error(pred, gt) / math.sqrt(ga)


class Timer:
    """Context manager measuring elapsed milliseconds via OpenCV's tick counter.

    On Windows ``cv.getTickFrequency()`` is the QPC frequency (~1e7 Hz, ~100 ns
    resolution), which is ample for the 1-50 ms updates being measured.
    """

    __slots__ = ("_start", "ms")

    def __init__(self) -> None:
        self._start = 0
        self.ms = 0.0

    def __enter__(self) -> Timer:
        self._start = cv.getTickCount()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (cv.getTickCount() - self._start) / cv.getTickFrequency() * 1000.0


@dataclass(slots=True)
class FrameRecord:
    index: int
    pred: BBox | None
    gt: BBox | None
    ok: bool
    confidence: float | None
    iou: float
    centre_err: float
    scale_ratio: float
    update_ms: float
    gt_valid: bool = True
    gt_absent: bool = False


@dataclass
class RunStats:
    tracker: str
    type: str
    video: str
    detector: str | None = None
    n_frames: int = 0
    n_eval: int = 0
    mean_iou: float = float("nan")
    median_iou: float = float("nan")
    success_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    success_auc: float = float("nan")
    precision_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    precision_20: float = float("nan")
    mean_centre_err: float = float("nan")
    median_centre_err: float = float("nan")
    median_scale_ratio: float = float("nan")
    scale_err: float = float("nan")
    scale_err_p90: float = float("nan")
    frames_to_failure: int = -1
    time_to_failure_s: float = float("nan")
    failure_rate: float = float("nan")
    n_reacquisitions: int = 0
    init_ms: float = float("nan")
    fps_mean: float = float("nan")
    fps_median: float = float("nan")
    update_ms_median: float = float("nan")
    update_ms_p95: float = float("nan")

    def to_row(self) -> dict[str, str]:
        f2f = "-" if self.frames_to_failure < 0 else str(self.frames_to_failure)
        return {
            "Tracker": self.tracker,
            "Type": self.type,
            "Clip": Path(self.video).stem,
            "Frames": str(self.n_eval),
            "IoU": f"{self.mean_iou:.3f}",
            "AUC": f"{self.success_auc:.3f}",
            "Prec@20": f"{self.precision_20:.3f}",
            "Scale x": f"{self.median_scale_ratio:.3f}",
            "ScaleErr": f"{self.scale_err:.3f}",
            "F2F": f2f,
            "Fail%": f"{100.0 * self.failure_rate:.1f}",
            "FPS": f"{self.fps_median:.0f}",
            "ms/f": f"{self.update_ms_median:.2f}",
        }


DEFAULT_COLUMNS = [
    "Tracker", "Type", "Clip", "Frames", "IoU", "AUC", "Prec@20",
    "Scale x", "ScaleErr", "F2F", "Fail%", "FPS", "ms/f",
]
HEADLINE_COLUMNS = ["Tracker", "IoU", "ScaleErr", "FPS"]


def _first_sustained_failure(ious: np.ndarray, patience: int) -> int:
    """Index of the first IoU dip below threshold sustained for ``patience`` frames.

    The patience is not optional.  With the source clip's measured 140 px/frame
    jumps a single-frame dropout is routine, and an unguarded definition reports
    "failed at frame 1" for every tracker.
    """
    run = 0
    for i, value in enumerate(ious):
        if np.isnan(value):
            continue
        if value < FAILURE_IOU:
            run += 1
            if run >= patience:
                return i - patience + 1
        else:
            run = 0
    return -1


def _count_reacquisitions(ious: np.ndarray) -> int:
    """Times IoU climbs back above REACQUIRE_IOU after a sustained failure.

    A tracker that fails at frame 10 and recovers is not summarised by a single
    frames-to-failure integer, so this is reported alongside it.
    """
    count = 0
    failed = False
    run = 0
    for value in ious:
        if np.isnan(value):
            continue
        if value < FAILURE_IOU:
            run += 1
            if run >= FAILURE_PATIENCE:
                failed = True
        else:
            run = 0
            if failed and value >= REACQUIRE_IOU:
                count += 1
                failed = False
    return count


def summarise(
    records: list[FrameRecord],
    *,
    tracker: str,
    type: str,
    video: str,
    detector: str | None = None,
    fps_hint: float = 30.0,
    fail_patience: int = FAILURE_PATIENCE,
) -> RunStats:
    stats = RunStats(tracker=tracker, type=type, video=str(video), detector=detector)
    stats.n_frames = len(records)
    if not records:
        return stats

    # Frames with no valid ground truth are excluded from every accuracy metric
    # and n_eval reports the honest denominator.  Absent (fully occluded) frames
    # are excluded too: no box is the correct answer there, and scoring them
    # would reward whichever tracker happens to freeze in the right place.
    evaluable = [r for r in records if r.gt_valid and not r.gt_absent and r.gt is not None]
    stats.n_eval = len(evaluable)

    ious = np.array([r.iou for r in evaluable], dtype=np.float64)
    centres = np.array([r.centre_err for r in evaluable], dtype=np.float64)
    ratios = np.array([r.scale_ratio for r in evaluable], dtype=np.float64)

    if stats.n_eval:
        finite_iou = ious[~np.isnan(ious)]
        if finite_iou.size:
            stats.mean_iou = float(finite_iou.mean())
            stats.median_iou = float(np.median(finite_iou))
            stats.success_curve = np.array(
                [float((finite_iou >= t - THRESHOLD_EPS).mean()) for t in SUCCESS_THRESHOLDS]
            )
            stats.success_auc = float(stats.success_curve.mean())
            stats.failure_rate = float((finite_iou < FAILURE_IOU).mean())

        finite_centre = centres[~np.isnan(centres)]
        if finite_centre.size:
            stats.mean_centre_err = float(finite_centre.mean())
            stats.median_centre_err = float(np.median(finite_centre))
            stats.precision_curve = np.array(
                [float((finite_centre <= t).mean()) for t in PRECISION_THRESHOLDS]
            )
            stats.precision_20 = float((finite_centre <= 20.0).mean())

        finite_ratio = ratios[(~np.isnan(ratios)) & (ratios > 0)]
        if finite_ratio.size:
            stats.median_scale_ratio = float(np.median(finite_ratio))
            octaves = np.abs(np.log2(finite_ratio))
            stats.scale_err = float(octaves.mean())
            stats.scale_err_p90 = float(np.percentile(octaves, 90))

        idx = _first_sustained_failure(ious, fail_patience)
        if idx >= 0:
            stats.frames_to_failure = int(evaluable[idx].index)
            stats.time_to_failure_s = stats.frames_to_failure / max(fps_hint, 1e-9)
        stats.n_reacquisitions = _count_reacquisitions(ious)

    # Timing excludes frame 0: correlation-filter trackers do heavy FFT setup on
    # init and would otherwise poison the mean.
    updates = np.array([r.update_ms for r in records[1:] if r.update_ms > 0], dtype=np.float64)
    if records and records[0].update_ms > 0:
        stats.init_ms = float(records[0].update_ms)
    if updates.size:
        stats.update_ms_median = float(np.median(updates))
        stats.update_ms_p95 = float(np.percentile(updates, 95))
        stats.fps_mean = float(1000.0 / updates.mean()) if updates.mean() > 0 else float("inf")
        stats.fps_median = (
            float(1000.0 / stats.update_ms_median) if stats.update_ms_median > 0 else float("inf")
        )
    return stats


def format_table(stats: list[RunStats], columns: list[str] | None = None) -> str:
    columns = columns or DEFAULT_COLUMNS
    rows = [s.to_row() for s in stats]
    widths = {
        c: max(len(c), *(len(r.get(c, "")) for r in rows)) if rows else len(c) for c in columns
    }
    lines = ["  ".join(c.rjust(widths[c]) for c in columns)]
    lines.append("  ".join("-" * widths[c] for c in columns))
    lines.extend("  ".join(r.get(c, "").rjust(widths[c]) for c in columns) for r in rows)
    return "\n".join(lines)


def write_summary_csv(stats: list[RunStats], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f for f in asdict(stats[0]) if not f.endswith("_curve")] if stats else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for s in stats:
            writer.writerow({k: v for k, v in asdict(s).items() if k in fields})


def write_frames_csv(records: list[FrameRecord], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame", "px", "py", "pw", "ph", "gx", "gy", "gw", "gh",
             "ok", "confidence", "iou", "centre_err", "scale_ratio", "update_ms",
             "gt_valid", "gt_absent"]
        )
        for r in records:
            p = r.pred if r.pred is not None else ("", "", "", "")
            g = r.gt if r.gt is not None else ("", "", "", "")
            writer.writerow(
                [r.index, *[f"{v:.3f}" if v != "" else "" for v in p],
                 *[f"{v:.3f}" if v != "" else "" for v in g],
                 int(r.ok),
                 "" if r.confidence is None else f"{r.confidence:.4f}",
                 f"{r.iou:.4f}", f"{r.centre_err:.3f}", f"{r.scale_ratio:.4f}",
                 f"{r.update_ms:.4f}", int(r.gt_valid), int(r.gt_absent)]
            )


def _plot_grid(series: list[tuple[str, np.ndarray]], width: int, height: int,
               y_lo: float, y_hi: float) -> list[str]:
    marks = "*+ox#@"
    grid = [[" "] * width for _ in range(height)]
    span = max(y_hi - y_lo, 1e-9)
    for s_i, (_, values) in enumerate(series):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            continue
        for col in range(width):
            lo = int(col * values.size / width)
            hi = max(lo + 1, int((col + 1) * values.size / width))
            chunk = values[lo:hi]
            chunk = chunk[~np.isnan(chunk)]
            if chunk.size == 0:
                continue
            y = float(chunk.mean())
            row = int(round((1.0 - (y - y_lo) / span) * (height - 1)))
            row = min(max(row, 0), height - 1)
            grid[row][col] = marks[s_i % len(marks)]
    return ["".join(r) for r in grid]


def ascii_series_plot(series: list[tuple[str, np.ndarray]], *, width: int = 60,
                      height: int = 12, label: str = "", y_lo: float | None = None,
                      y_hi: float | None = None) -> str:
    """Plot one or more series against frame index. No matplotlib available."""
    if not series:
        return ""
    all_values = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for _, v in series])
    all_values = all_values[~np.isnan(all_values)]
    if all_values.size == 0:
        return ""
    lo = float(all_values.min()) if y_lo is None else y_lo
    hi = float(all_values.max()) if y_hi is None else y_hi
    if hi - lo < 1e-9:
        hi = lo + 1.0

    rows = _plot_grid(series, width, height, lo, hi)
    out = [label] if label else []
    for i, row in enumerate(rows):
        tick = hi if i == 0 else (lo if i == len(rows) - 1 else None)
        prefix = f"{tick:7.2f} |" if tick is not None else " " * 7 + " |"
        out.append(prefix + row)
    out.append(" " * 7 + " +" + "-" * width)
    marks = "*+ox#@"
    out.append(" " * 9 + "  ".join(f"{marks[i % len(marks)]} {n}" for i, (n, _) in enumerate(series)))
    return "\n".join(out)


def ascii_success_plot(stats: list[RunStats], *, width: int = 60, height: int = 15) -> str:
    series = [(s.tracker, s.success_curve) for s in stats if s.success_curve.size]
    if not series:
        return ""
    return ascii_series_plot(
        series, width=width, height=height,
        label="Success plot (x = IoU threshold 0..1, y = fraction of frames)",
        y_lo=0.0, y_hi=1.0,
    )
