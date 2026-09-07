"""Per-frame ground truth for the tracking bench.

Two sources, in priority order:

1. A generator sidecar ``<stem>.gt.json``.  Exact, sub-pixel, and the *only*
   correct ground truth for occlusion clips -- thresholding cannot recover the
   box of a target that is hidden, by definition.
2. Threshold + contour extraction via :class:`~trackers.detectors.BlobDetector`.
   Mandatory regardless, because ``tracking_test_video.mp4`` has no sidecar and
   never will.

Running (2) against a clip that has (1) is a free self-test of the extractor,
and that cross-check is part of the build order.

Selection policy
----------------
Candidates come from ``BlobDetector(thresh=200, min_area=200, min_extent=0.5)``.
Measured on ``tracking_test_video.mp4`` this yields exactly one candidate on
**2834 of 2835** frames and zero on one.

The ~70 frames that a naive ``area > 20`` filter flags as ambiguous (runs
1393-1449 and 2822-2834) are *not* the circle clipped at a frame edge -- a large
blob touches a border on 0 of 2835 frames, and the target stays within
x in [60, 1214], y in [16, 634].  The second blob is a static arrow/pointer
graphic at ~(553, 274, 69, 60): area ~57 px^2, extent ~0.014, against the
circle's ~910 px^2 and extent ~0.72.  The extent predicate removes it outright,
so no tie-break is normally needed.

``policy`` handles anything that still gets through:

``"largest"`` (default)
    Greatest contour area wins.
``"nearest"``
    Nearest centre to the previously accepted frame.  Correct if a same-size
    distractor ever appears.
``"strict"``
    Mark the frame invalid unless exactly one candidate survives.  Use when
    auditing a new clip.

Frames with no candidate are ``valid=False, bbox=None``.  They are **skipped**,
never interpolated: metrics exclude them and report ``n_eval`` so the
denominator stays honest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2 as cv
import numpy as np

from .detectors import BlobDetector, Detector
from .types import BBox, bbox_centre

__all__ = [
    "GTFrame",
    "GroundTruth",
    "cache_path_for",
    "extract_from_video",
    "load",
    "load_sidecar",
    "sidecar_path_for",
]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GTFrame:
    index: int
    bbox: BBox | None
    valid: bool
    absent: bool = False
    n_candidates: int = 0
    source: str = "blob"
    visible: float = 1.0


class GroundTruth:
    """Per-frame ground truth, indexable by frame number."""

    def __init__(
        self,
        frames: list[GTFrame],
        width: int,
        height: int,
        fps: float,
        source: str,
        meta: dict | None = None,
    ) -> None:
        self.frames = frames
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.source = source
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> GTFrame:
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return GTFrame(index=index, bbox=None, valid=False, source=self.source)

    def valid_indices(self) -> np.ndarray:
        return np.array([f.index for f in self.frames if f.valid], dtype=np.int64)

    def first_valid(self, start: int = 0) -> GTFrame:
        for frame in self.frames[start:]:
            if frame.valid:
                return frame
        raise ValueError("ground truth contains no valid frame")

    def as_array(self) -> np.ndarray:
        """(N, 4) float array of boxes; NaN rows where invalid."""
        out = np.full((len(self.frames), 4), np.nan, dtype=np.float64)
        for i, frame in enumerate(self.frames):
            if frame.valid and frame.bbox is not None:
                out[i] = frame.bbox
        return out

    def stats(self) -> dict[str, float | int]:
        boxes = self.as_array()
        valid = ~np.isnan(boxes[:, 0])
        centres = np.column_stack(
            [boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3] / 2.0]
        )
        both = valid[:-1] & valid[1:]
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)[both] if len(centres) > 1 else np.array([])

        out: dict[str, float | int] = {
            "n_frames": len(self.frames),
            "n_valid": int(valid.sum()),
            "n_invalid": int((~valid).sum()),
            "n_absent": int(sum(1 for f in self.frames if f.absent)),
        }
        if valid.any():
            out |= {
                "min_w": float(np.nanmin(boxes[:, 2])),
                "max_w": float(np.nanmax(boxes[:, 2])),
                "mean_w": float(np.nanmean(boxes[:, 2])),
                "x_min": float(np.nanmin(boxes[:, 0])),
                "x_max": float(np.nanmax(boxes[:, 0] + boxes[:, 2])),
                "y_min": float(np.nanmin(boxes[:, 1])),
                "y_max": float(np.nanmax(boxes[:, 1] + boxes[:, 3])),
            }
        if steps.size:
            out |= {
                "mean_step_px": float(steps.mean()),
                "median_step_px": float(np.median(steps)),
                "max_step_px": float(steps.max()),
                "p95_step_px": float(np.percentile(steps, 95)),
            }
        return out


def sidecar_path_for(video: Path | str) -> Path:
    video = Path(video)
    return video.with_suffix("").with_suffix(".gt.json") if video.suffix else video


def cache_path_for(video: Path | str) -> Path:
    video = Path(video)
    return video.parent / (video.stem + ".gt.cache.json")


def _sidecar_candidates(video: Path) -> list[Path]:
    return [video.parent / (video.stem + ".gt.json")]


def load_sidecar(path: Path | str) -> GroundTruth:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [
        GTFrame(
            index=int(row["i"]),
            bbox=tuple(float(v) for v in row["bbox"]),  # type: ignore[arg-type]
            valid=True,
            absent=bool(row.get("absent", False)),
            n_candidates=1,
            source="sidecar",
            visible=float(row.get("visible", 1.0)),
        )
        for row in data["frames"]
    ]
    return GroundTruth(
        frames=frames,
        width=int(data["width"]),
        height=int(data["height"]),
        fps=float(data["fps"]),
        source="sidecar",
        meta={k: v for k, v in data.items() if k != "frames"},
    )


def _select(candidates, policy: str, prev_centre: tuple[float, float] | None):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if policy == "strict":
        return None
    if policy == "nearest" and prev_centre is not None:
        return min(
            candidates,
            key=lambda d: math.dist(bbox_centre(d.bbox), prev_centre),
        )
    # "largest" (default), and the fallback for "nearest" on the first frame.
    return max(candidates, key=lambda d: d.bbox[2] * d.bbox[3])


def extract_from_video(
    video: Path | str,
    *,
    detector: Detector | None = None,
    policy: str = "largest",
    progress: bool = False,
) -> GroundTruth:
    """Extract ground truth by running ``detector`` over every frame.

    Frames are read sequentially with ``grab``/``retrieve``.  Neither
    ``CAP_PROP_FRAME_COUNT`` nor ``CAP_PROP_POS_FRAMES`` is trusted: on the
    source clip, seeking to 2834 returns frame 2833's content and the reported
    count is one more than the number of decodable frames.
    """
    video = Path(video)
    if policy not in {"largest", "nearest", "strict"}:
        raise ValueError(f"unknown policy {policy!r}")

    detector = detector or BlobDetector()
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv.CAP_PROP_FPS)) or 30.0

    frames: list[GTFrame] = []
    prev_centre: tuple[float, float] | None = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            candidates = detector.detect(frame)
            chosen = _select(candidates, policy, prev_centre)
            if chosen is None:
                frames.append(
                    GTFrame(index, None, False, n_candidates=len(candidates), source="blob")
                )
            else:
                frames.append(
                    GTFrame(
                        index,
                        chosen.bbox,
                        True,
                        n_candidates=len(candidates),
                        source="blob",
                    )
                )
                prev_centre = bbox_centre(chosen.bbox)
            index += 1
            if progress and index % 500 == 0:
                print(f"  ...{index} frames", file=sys.stderr)
    finally:
        cap.release()

    return GroundTruth(frames, width, height, fps, source="blob")


def _params_hash(detector: Detector, policy: str) -> str:
    params = getattr(detector, "params", lambda: {"name": getattr(detector, "name", "?")})()
    payload = json.dumps({"detector": params, "policy": policy}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_cache(path: Path, key: dict) -> GroundTruth | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema") != SCHEMA_VERSION or data.get("key") != key:
        return None
    frames = [
        GTFrame(
            index=int(row[0]),
            bbox=(row[1], row[2], row[3], row[4]) if row[1] is not None else None,
            valid=row[1] is not None,
            n_candidates=int(row[5]),
            source="blob",
        )
        for row in data["frames"]
    ]
    return GroundTruth(frames, data["width"], data["height"], data["fps"], source="blob")


def _write_cache(path: Path, key: dict, gt: GroundTruth) -> None:
    rows = [
        [f.index, *(f.bbox if f.bbox is not None else (None, None, None, None)), f.n_candidates]
        for f in gt.frames
    ]
    payload = {
        "schema": SCHEMA_VERSION,
        "key": key,
        "width": gt.width,
        "height": gt.height,
        "fps": gt.fps,
        "frames": rows,
    }
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # a cache we cannot write is not an error


def load(
    video: Path | str,
    *,
    prefer_sidecar: bool = True,
    use_cache: bool = True,
    detector: Detector | None = None,
    policy: str = "largest",
    progress: bool = False,
) -> GroundTruth:
    """Load ground truth for ``video``: sidecar if present, else extraction."""
    video = Path(video)
    if prefer_sidecar:
        for candidate in _sidecar_candidates(video):
            if candidate.exists():
                return load_sidecar(candidate)

    detector = detector or BlobDetector()
    stat = video.stat()
    key = {
        "mtime": int(stat.st_mtime),
        "size": stat.st_size,
        "params": _params_hash(detector, policy),
    }
    cache = cache_path_for(video)
    if use_cache:
        cached = _read_cache(cache, key)
        if cached is not None:
            return cached

    gt = extract_from_video(video, detector=detector, policy=policy, progress=progress)
    if use_cache:
        _write_cache(cache, key, gt)
    return gt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--report", action="store_true", help="print a census and displacement stats")
    parser.add_argument("--policy", default="largest", choices=["largest", "nearest", "strict"])
    parser.add_argument("--thresh", type=int, default=200)
    parser.add_argument("--min-area", type=float, default=200.0)
    parser.add_argument("--min-extent", type=float, default=0.5)
    parser.add_argument("--no-sidecar", action="store_true", help="force threshold extraction")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    detector = BlobDetector(
        thresh=args.thresh, min_area=args.min_area, min_extent=args.min_extent
    )
    gt = load(
        args.video,
        prefer_sidecar=not args.no_sidecar,
        use_cache=not args.no_cache,
        detector=detector,
        policy=args.policy,
        progress=True,
    )

    stats = gt.stats()
    invalid = [f.index for f in gt.frames if not f.valid]
    multi = [f.index for f in gt.frames if f.n_candidates > 1]

    print(f"{args.video}  ({gt.width}x{gt.height} @ {gt.fps:.2f} fps, source={gt.source})")
    print(f"  valid {stats['n_valid']} / {stats['n_frames']}, invalid {stats['n_invalid']}")
    if invalid:
        print(f"  invalid frames: {invalid if len(invalid) <= 12 else invalid[:12] + ['...']}")
    if multi:
        print(f"  frames with >1 candidate: {len(multi)}")
    if args.report and stats.get("mean_step_px") is not None:
        print(f"  box width   min/mean/max: {stats['min_w']:.1f} / {stats['mean_w']:.1f} / {stats['max_w']:.1f}")
        print(f"  box extent  x [{stats['x_min']:.0f}, {stats['x_max']:.0f}]  y [{stats['y_min']:.0f}, {stats['y_max']:.0f}]")
        print(
            f"  displacement px/frame  mean {stats['mean_step_px']:.1f}"
            f"  median {stats['median_step_px']:.1f}"
            f"  p95 {stats['p95_step_px']:.1f}"
            f"  max {stats['max_step_px']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
