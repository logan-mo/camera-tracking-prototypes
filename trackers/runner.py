"""Run one tracker over one video and measure it.

Timing discipline: the ``Timer`` wraps **only** ``tracker.update(frame)``.
Frame reads, colour conversion, drawing, ``imshow``/``waitKey``, video writing
and metric computation are all outside it.  ``--headless`` skips drawing and
``waitKey`` entirely, because ``waitKey(1)`` alone dominates wall-clock over a
few thousand frames and would otherwise be reported as tracker cost.

Frames are read sequentially with ``grab``/``retrieve``; ``CAP_PROP_POS_FRAMES``
is never used for stepping.  On the source clip, seeking to frame 2834 returns
frame 2833's content, so seeks are not frame-exact here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from . import groundtruth as gt_mod
from . import registry
from .detectors import detector_from_spec
from .metrics import (
    FrameRecord,
    RunStats,
    Timer,
    centre_error,
    format_table,
    iou,
    scale_ratio,
    summarise,
    write_frames_csv,
)
from .types import BBox, bbox_centre

__all__ = ["RunConfig", "RunResult", "draw_overlay", "main", "run"]

COLOUR_GT = (80, 220, 80)
COLOUR_PRED = (220, 80, 220)
COLOUR_LOST = (60, 60, 235)
COLOUR_HUD = (235, 235, 235)


@dataclass
class RunConfig:
    video: Path
    tracker: str
    detector: str = "blob"
    start_frame: int = 0
    max_frames: int | None = None
    init_bbox: BBox | None = None
    select_roi: bool = False
    show: bool = True
    save: Path | None = None
    gt_policy: str = "largest"
    prefer_sidecar: bool = True
    fail_patience: int = 3
    draw_trail: bool = True
    delay_ms: int = 1
    tracker_kwargs: dict = field(default_factory=dict)


@dataclass
class RunResult:
    stats: RunStats
    records: list[FrameRecord]


def _resolve_init_bbox(cfg: RunConfig, gt: gt_mod.GroundTruth, frame: np.ndarray,
                       index: int) -> tuple[BBox, int]:
    """Priority: explicit --bbox, then --select-roi, then ground truth.

    Ground truth is the default because it is deterministic; a hand-drawn ROI
    makes runs incomparable between sessions, which defeats the benchmark.
    """
    if cfg.init_bbox is not None:
        return cfg.init_bbox, index
    if cfg.select_roi:
        roi = cv.selectROI("select target (ENTER to confirm)", frame, showCrosshair=True)
        cv.destroyWindow("select target (ENTER to confirm)")
        # selectROI returns an all-zero rect when the user presses ESC.
        if roi[2] > 0 and roi[3] > 0:
            return (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])), index
        print("selectROI cancelled; falling back to ground truth", file=sys.stderr)

    entry = gt[index]
    if entry.valid and entry.bbox is not None:
        return entry.bbox, index
    nxt = gt.first_valid(index)
    print(
        f"ground truth invalid at frame {index}; starting at {nxt.index} instead",
        file=sys.stderr,
    )
    return nxt.bbox, nxt.index  # type: ignore[return-value]


def _to_int_rect(bbox: BBox) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def draw_overlay(
    frame: np.ndarray, *, pred: BBox | None, gt: BBox | None, ok: bool,
    conf: float | None, fps: float, name: str, idx: int, n: int | None,
    frame_iou: float, sr: float, trail: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    if gt is not None:
        x, y, w, h = _to_int_rect(gt)
        cv.rectangle(canvas, (x, y), (x + w, y + h), COLOUR_GT, 2)
    if pred is not None:
        x, y, w, h = _to_int_rect(pred)
        cv.rectangle(canvas, (x, y), (x + w, y + h), COLOUR_PRED if ok else COLOUR_LOST, 2)
        # A line joining the two centres makes position error and size error
        # legible at a glance, which a pair of rectangles alone does not.
        if gt is not None:
            pc = tuple(int(round(v)) for v in bbox_centre(pred))
            gc = tuple(int(round(v)) for v in bbox_centre(gt))
            cv.line(canvas, pc, gc, COLOUR_HUD, 1)
    if trail:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in trail], dtype=np.int32)
        cv.polylines(canvas, [pts], False, COLOUR_PRED, 1)

    total = f"/{n:04d}" if n else ""
    hud = (
        f"{name} | f {idx:04d}{total} | {fps:6.1f} fps"
        f" | IoU {frame_iou:.2f} | scale {sr:.2f}"
    )
    if conf is not None and np.isfinite(conf):
        hud += f" | conf {conf:.2f}"
    cv.putText(canvas, hud, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(canvas, hud, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_HUD, 1, cv.LINE_AA)
    return canvas


def run(cfg: RunConfig) -> RunResult:
    video = Path(cfg.video)
    gt = gt_mod.load(video, prefer_sidecar=cfg.prefer_sidecar, policy=cfg.gt_policy)

    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps_hint = float(cap.get(cv.CAP_PROP_FPS)) or 30.0

    # Sequential skipping -- seeking is not frame-exact on these files.
    index = 0
    while index < cfg.start_frame:
        if not cap.grab():
            cap.release()
            raise RuntimeError(f"video ended before start frame {cfg.start_frame}")
        index += 1

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("could not read the first frame")

    init_bbox, init_index = _resolve_init_bbox(cfg, gt, frame, index)
    while index < init_index:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("video ended while seeking a valid init frame")
        index += 1

    kwargs = dict(cfg.tracker_kwargs)
    if registry.needs_groundtruth(cfg.tracker):
        kwargs["gt"] = gt
    if registry.needs_detector(cfg.tracker):
        kwargs["detector"] = detector_from_spec(cfg.detector)
    tracker = registry.build(cfg.tracker, **kwargs)

    writer: cv.VideoWriter | None = None
    if cfg.save is not None:
        cfg.save.parent.mkdir(parents=True, exist_ok=True)
        writer = cv.VideoWriter(
            str(cfg.save), cv.VideoWriter_fourcc(*"mp4v"), fps_hint,
            (frame.shape[1], frame.shape[0]),
        )

    records: list[FrameRecord] = []
    trail: list[tuple[float, float]] = []
    draw = cfg.show or writer is not None

    try:
        with Timer() as timer:
            tracker.init(frame, init_bbox)
        entry = gt[index]
        records.append(
            FrameRecord(
                index=index, pred=init_bbox, gt=entry.bbox, ok=True,
                confidence=getattr(tracker, "confidence", None),
                iou=iou(init_bbox, entry.bbox),
                centre_err=centre_error(init_bbox, entry.bbox),
                scale_ratio=scale_ratio(init_bbox, entry.bbox),
                update_ms=timer.ms, gt_valid=entry.valid, gt_absent=entry.absent,
            )
        )

        n_done = 1
        while cfg.max_frames is None or n_done < cfg.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1

            with Timer() as timer:
                tracked, bbox = tracker.update(frame)

            entry = gt[index]
            frame_iou = iou(bbox, entry.bbox) if entry.valid else float("nan")
            sr = scale_ratio(bbox, entry.bbox) if entry.valid else float("nan")
            conf = getattr(tracker, "confidence", None)
            records.append(
                FrameRecord(
                    index=index, pred=bbox, gt=entry.bbox, ok=bool(tracked), confidence=conf,
                    iou=frame_iou, centre_err=centre_error(bbox, entry.bbox) if entry.valid else float("nan"),
                    scale_ratio=sr, update_ms=timer.ms,
                    gt_valid=entry.valid, gt_absent=entry.absent,
                )
            )

            if draw:
                if cfg.draw_trail:
                    trail.append(bbox_centre(bbox))
                    if len(trail) > 60:
                        trail.pop(0)
                inst_fps = 1000.0 / timer.ms if timer.ms > 0 else float("inf")
                canvas = draw_overlay(
                    frame, pred=bbox, gt=entry.bbox, ok=bool(tracked), conf=conf,
                    fps=inst_fps, name=getattr(tracker, "name", cfg.tracker),
                    idx=index, n=len(gt), frame_iou=frame_iou, sr=sr,
                    trail=trail if cfg.draw_trail else None,
                )
                if writer is not None:
                    writer.write(canvas)
                if cfg.show:
                    cv.imshow("track", canvas)
                    if (cv.waitKey(cfg.delay_ms) & 0xFF) == 27:
                        break
            n_done += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.show:
            cv.destroyAllWindows()

    stats = summarise(
        records,
        tracker=getattr(tracker, "name", cfg.tracker),
        type=getattr(tracker, "type", registry.tracker_type(cfg.tracker)),
        video=str(video),
        detector=cfg.detector if registry.needs_detector(cfg.tracker) else None,
        fps_hint=fps_hint,
        fail_patience=cfg.fail_patience,
    )
    return RunResult(stats=stats, records=records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one tracker over one video.")
    parser.add_argument("video", type=Path, nargs="?", default=Path("tracking_test_video.mp4"))
    parser.add_argument("--tracker", "-t", default="klt",
                        help=f"one of: {', '.join(sorted(registry.TRACKER_SPECS))}")
    parser.add_argument("--detector", default="blob")
    parser.add_argument("--bbox", help="x,y,w,h init box (overrides ground truth)")
    parser.add_argument("--select-roi", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-trail", action="store_true")
    parser.add_argument("--delay", type=int, default=1, help="waitKey delay, ms")
    parser.add_argument("--csv", type=Path, help="write the per-frame record here")
    parser.add_argument("--no-sidecar", action="store_true")
    args = parser.parse_args(argv)

    init_bbox = None
    if args.bbox:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            parser.error("--bbox needs exactly x,y,w,h")
        init_bbox = tuple(parts)  # type: ignore[assignment]

    cfg = RunConfig(
        video=args.video, tracker=args.tracker, detector=args.detector,
        start_frame=args.start_frame, max_frames=args.max_frames,
        init_bbox=init_bbox, select_roi=args.select_roi,
        show=not args.headless, save=args.save,
        prefer_sidecar=not args.no_sidecar,
        draw_trail=not args.no_trail, delay_ms=args.delay,
    )
    try:
        result = run(cfg)
    except ImportError as exc:
        print(f"{args.tracker}: not implemented yet ({exc})", file=sys.stderr)
        return 2

    print(format_table([result.stats]))
    if args.csv:
        write_frames_csv(result.records, args.csv)
        print(f"per-frame records -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
