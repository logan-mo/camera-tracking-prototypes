"""Generate synthetic tracking clips with exact ground truth.

Why this exists
---------------
``tracking_test_video.mp4`` has two properties that make it useless for the
comparison it was meant to support:

* The circle is 34-36 px wide on **every** one of its 2835 frames, so it cannot
  exercise the scale-variation failure mode that puts DSST and CSRT in scope.
* Its target moves a median of 24.6 px/frame -- 0.7 of its own diameter -- and up
  to 140 px.  A correlation filter follows roughly half a search window per
  frame, so no padding choice tracks it (measured best: 56.5% of frames within
  20 px, at 4x the FFT area of a standard window).

The clips generated here fix both.  Speed caps are the point: at 5 px/frame
translation is a non-issue and **scale is the only variable**, so the benchmark
measures scale handling rather than search-window size.  ``synth_fast``
deliberately reproduces the source clip's regime so the report can *quantify*
why that clip is hard rather than hand-wave it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

# The source clip renders its background at exactly this value, not black.
BG_VALUE = 15
FG_VALUE = 255

# Occluders are mid-grey by default, never white.  White bars would merge with
# the target in CSRT's foreground/background histogram model, accidentally
# designing the one scenario where a spatial reliability map cannot help.
OCCLUDER_VALUE = 128

# cv.circle sub-pixel shift: coordinates are in 1/2**SHIFT pixel units.
SHIFT = 3
SUB = 1 << SHIFT

# Half-pixel convention.  OpenCV drawing and cv.boundingRect treat an integer
# coordinate as a pixel *centre*: a disc of radius r centred on pixel 100 covers
# pixels 83..117, and boundingRect reports (x=83, w=35).  The bench's BBox
# convention instead treats pixel i as covering the continuous span [i, i+1), so
# that box is continuous [83, 118) with centre 100.5 = x + w/2.
#
# Both are self-consistent; mixing them costs exactly half a pixel.  The
# conversion happens here, once, at the only point where pixel-centre
# coordinates enter recorded ground truth -- which keeps `cx = x + w/2` valid
# everywhere else, including under DSST's float scale factors.
#
# Caught by the generator/extractor cross-check as a +0.50 px bias in both axes
# (against 0.15-0.23 px of genuine antialiasing scatter).
PIXEL_CENTRE_OFFSET = 0.5

# DSST searches a discrete scale pool at this step.  A clip whose per-frame
# radius ratio exceeds it measures the clip, not the tracker.
SCALE_POOL_STEP = 1.02


@dataclass
class ClipSpec:
    name: str
    frames: int = 900
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    speed_max: float = 5.0
    radius: float = 17.0
    radius_min: float | None = None
    radius_max: float | None = None
    scale_period: float = 360.0
    occluders: list[tuple[int, int, int, int]] = field(default_factory=list)
    occluder_value: int = OCCLUDER_VALUE
    seed: int = 1234
    turn_sigma: float = 0.12
    speed_lag: float = 0.05
    regime_frames: int = 90
    margin: int = 8

    @property
    def varies_scale(self) -> bool:
        return self.radius_min is not None and self.radius_max is not None


PRESETS: dict[str, ClipSpec] = {
    "baseline": ClipSpec(name="synth_baseline", speed_max=4.0, radius=17.0),
    "scale": ClipSpec(
        name="synth_scale", speed_max=5.0, radius_min=12.0, radius_max=48.0, scale_period=360.0
    ),
    "occlusion": ClipSpec(
        name="synth_occlusion",
        speed_max=6.0,
        radius=17.0,
        occluders=[(320, 0, 60, 720), (640, 0, 60, 720), (960, 0, 60, 720)],
    ),
    # speed_max is tuned so the realised displacement matches the source clip's
    # measured regime (mean 28.5, median 24.6, max 140.5 px/frame).  The stop
    # regimes are kept: the source clip is stationary 50% of the time in some
    # stretches and 0% in others, and reproducing that spread is the point.
    "fast": ClipSpec(name="synth_fast", speed_max=85.0, radius=17.0),
    "scale_occlusion": ClipSpec(
        name="synth_scale_occlusion",
        speed_max=5.0,
        radius_min=12.0,
        radius_max=48.0,
        scale_period=360.0,
        occluders=[(320, 0, 60, 720), (960, 0, 60, 720)],
    ),
}


def radius_schedule(spec: ClipSpec) -> np.ndarray:
    """Per-frame radius.

    For scale clips: ``r(t) = r_mid * exp(A * sin(2*pi*t/T))`` with phase 0, so
    frame 0 sits at mid-scale and the fixed-box failure is symmetric about the
    init size.  ``r_mid = sqrt(r_min * r_max)`` and ``A = ln(r_max/r_min)/2``.
    """
    t = np.arange(spec.frames, dtype=np.float64)
    if not spec.varies_scale:
        return np.full(spec.frames, float(spec.radius))
    r_min = float(spec.radius_min)  # type: ignore[arg-type]
    r_max = float(spec.radius_max)  # type: ignore[arg-type]
    r_mid = math.sqrt(r_min * r_max)
    amp = math.log(r_max / r_min) / 2.0
    return r_mid * np.exp(amp * np.sin(2.0 * np.pi * t / spec.scale_period))


def check_scale_rate(spec: ClipSpec, radii: np.ndarray) -> tuple[bool, float, float]:
    """Verify the per-frame radius ratio stays inside the DSST scale pool.

    ``max|d ln r / dt| = A * 2*pi / T`` must be below ``ln(1.02)``, i.e.
    ``T >= A * 2*pi / ln(1.02)``.  For A = ln(2) that is T >= 220; the default
    T = 360 gives a max per-frame ratio of 1.0123, comfortably inside the pool.
    """
    if len(radii) < 2:
        return True, 1.0, math.inf
    ratios = radii[1:] / radii[:-1]
    worst = float(max(ratios.max(), 1.0 / ratios.min()))
    if not spec.varies_scale:
        return True, worst, math.inf
    amp = math.log(float(spec.radius_max) / float(spec.radius_min)) / 2.0  # type: ignore[arg-type]
    min_period = amp * 2.0 * math.pi / math.log(SCALE_POOL_STEP)
    return worst < SCALE_POOL_STEP, worst, min_period


def simulate_motion(spec: ClipSpec, radii: np.ndarray) -> np.ndarray:
    """Heading/speed random walk, reflected off a margin-inset rectangle.

    Speed is lagged toward a piecewise-constant regime target rather than
    resampled every frame, which produces the smooth ramps and stop-and-go
    character the source clip shows (53% stationary in its first 200 frames)
    instead of Gaussian jitter.  Reflection keeps the disc fully inside the
    frame, matching the measured source clip -- a large blob touches a border on
    0 of its 2835 frames -- so ground truth is never ambiguous.
    """
    rng = np.random.default_rng(spec.seed)
    positions = np.zeros((spec.frames, 2), dtype=np.float64)

    r0 = float(radii[0])
    cx = rng.uniform(r0 + spec.margin, spec.width - r0 - spec.margin)
    cy = rng.uniform(r0 + spec.margin, spec.height - r0 - spec.margin)
    heading = rng.uniform(0.0, 2.0 * np.pi)
    speed = 0.0
    target_speed = rng.uniform(0.2 * spec.speed_max, spec.speed_max)

    for i in range(spec.frames):
        positions[i] = (cx, cy)

        if i > 0 and i % spec.regime_frames == 0:
            # A 1-in-5 chance of a full stop reproduces the source clip's pauses.
            target_speed = (
                0.0 if rng.random() < 0.2 else rng.uniform(0.2 * spec.speed_max, spec.speed_max)
            )

        heading += rng.normal(0.0, spec.turn_sigma)
        speed += (target_speed - speed) * spec.speed_lag
        cx += speed * math.cos(heading)
        cy += speed * math.sin(heading)

        r = float(radii[min(i + 1, spec.frames - 1)])
        lo_x, hi_x = r + spec.margin, spec.width - r - spec.margin
        lo_y, hi_y = r + spec.margin, spec.height - r - spec.margin
        if cx < lo_x:
            cx = 2 * lo_x - cx
            heading = math.pi - heading
        elif cx > hi_x:
            cx = 2 * hi_x - cx
            heading = math.pi - heading
        if cy < lo_y:
            cy = 2 * lo_y - cy
            heading = -heading
        elif cy > hi_y:
            cy = 2 * hi_y - cy
            heading = -heading
        cx = min(max(cx, lo_x), hi_x)
        cy = min(max(cy, lo_y), hi_y)

    return positions


def visible_fraction(cx: float, cy: float, r: float, spec: ClipSpec) -> float:
    """Fraction of the disc not covered by an occluder rectangle."""
    if not spec.occluders:
        return 1.0
    pad = int(math.ceil(r)) + 2
    size = 2 * pad + 1
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((xx - pad) ** 2 + (yy - pad) ** 2) <= r * r
    total = int(disc.sum())
    if total == 0:
        return 0.0

    hidden = np.zeros_like(disc)
    ox0, oy0 = int(round(cx)) - pad, int(round(cy)) - pad
    for rx, ry, rw, rh in spec.occluders:
        x0 = max(0, rx - ox0)
        y0 = max(0, ry - oy0)
        x1 = min(size, rx + rw - ox0)
        y1 = min(size, ry + rh - oy0)
        if x0 < x1 and y0 < y1:
            hidden[y0:y1, x0:x1] = True
    return float((disc & ~hidden).sum()) / float(total)


def render_frame(cx: float, cy: float, r: float, spec: ClipSpec) -> np.ndarray:
    frame = np.full((spec.height, spec.width, 3), BG_VALUE, dtype=np.uint8)
    cv.circle(
        frame,
        (int(round(cx * SUB)), int(round(cy * SUB))),
        int(round(r * SUB)),
        (FG_VALUE, FG_VALUE, FG_VALUE),
        -1,
        lineType=cv.LINE_AA,
        shift=SHIFT,
    )
    # Occluders are drawn after the circle so they cover it.
    for rx, ry, rw, rh in spec.occluders:
        cv.rectangle(
            frame,
            (rx, ry),
            (rx + rw - 1, ry + rh - 1),
            (spec.occluder_value,) * 3,
            -1,
        )
    return frame


CODEC_PREFS: list[tuple[str, str]] = [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]


def open_writer(path: Path, spec: ClipSpec, codec: str | None) -> tuple[cv.VideoWriter, Path]:
    """Open a VideoWriter, probing codecs until one actually works.

    Default is ``mp4v`` (MPEG-4 Part 2), which the bundled
    ``opencv_videoio_ffmpeg500_64.dll`` always has.  ``avc1``/``H264`` is
    deliberately not the default: OpenCV's FFmpeg wrapper defers H.264 to a
    separately downloaded ``openh264-*.dll``, and when it is missing the writer
    silently produces an unreadable file -- the most common Windows failure here.
    """
    prefs = [(codec, path.suffix or ".mp4")] if codec else CODEC_PREFS
    size = (spec.width, spec.height)
    errors: list[str] = []
    for fourcc_name, suffix in prefs:
        out_path = path.with_suffix(suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv.VideoWriter(
            str(out_path), cv.VideoWriter_fourcc(*fourcc_name), spec.fps, size
        )
        if writer.isOpened():
            return writer, out_path
        writer.release()
        errors.append(fourcc_name)
    raise RuntimeError(f"no working codec for {path} (tried {', '.join(errors)})")


def write_sidecar(path: Path, spec: ClipSpec, rows: list[dict], argv: list[str]) -> None:
    payload = {
        "schema": 1,
        "video": path.name,
        "clip": spec.name,
        "width": spec.width,
        "height": spec.height,
        "fps": spec.fps,
        "n_frames": len(rows),
        "seed": spec.seed,
        "generator": "tools/make_test_videos.py",
        "argv": argv,
        "bg_value": BG_VALUE,
        "fg_value": FG_VALUE,
        "occluders": [list(o) for o in spec.occluders],
        "occluder_value": spec.occluder_value,
        "frames": rows,
    }
    json_path = path.parent / (path.stem + ".gt.json")
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    csv_path = path.parent / (path.stem + ".gt.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "w", "h", "cx", "cy", "radius", "visible", "absent"])
        for row in rows:
            x, y, w, h = row["bbox"]
            writer.writerow(
                [
                    row["i"],
                    f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}",
                    f"{row['cx']:.3f}", f"{row['cy']:.3f}", f"{row['r']:.3f}",
                    f"{row['visible']:.4f}", int(row["absent"]),
                ]
            )


def generate(spec: ClipSpec, out: Path, *, codec: str | None = None,
             sidecar: bool = True, preview: bool = False, dry_run: bool = False,
             force: bool = False, argv: list[str] | None = None) -> Path:
    radii = radius_schedule(spec)
    ok, worst, min_period = check_scale_rate(spec, radii)
    if not ok and not force:
        raise SystemExit(
            f"{spec.name}: per-frame radius ratio {worst:.4f} exceeds the DSST scale pool "
            f"step {SCALE_POOL_STEP}. Raise --scale-period to at least {min_period:.0f} "
            f"(currently {spec.scale_period:.0f}), or pass --force to emit anyway."
        )

    positions = simulate_motion(spec, radii)

    rows: list[dict] = []
    for i in range(spec.frames):
        cx, cy = positions[i]
        r = float(radii[i])
        vis = visible_fraction(cx, cy, r, spec)
        # Convert the pixel-centre draw coordinate to the bench's continuous
        # convention; see PIXEL_CENTRE_OFFSET.
        ccx = float(cx) + PIXEL_CENTRE_OFFSET
        ccy = float(cy) + PIXEL_CENTRE_OFFSET
        rows.append(
            {
                "i": i,
                "bbox": [ccx - r, ccy - r, 2 * r, 2 * r],
                "cx": ccx,
                "cy": ccy,
                "r": r,
                "visible": vis,
                "absent": vis < 0.1,
            }
        )

    out_path = out
    if not dry_run:
        writer, out_path = open_writer(out, spec, codec)
        try:
            for i in range(spec.frames):
                cx, cy = positions[i]
                frame = render_frame(cx, cy, float(radii[i]), spec)
                writer.write(frame)
                if preview:
                    cv.imshow("make_test_videos", frame)
                    if (cv.waitKey(1) & 0xFF) == 27:
                        break
        finally:
            writer.release()
            if preview:
                cv.destroyAllWindows()

        size = out_path.stat().st_size if out_path.exists() else 0
        if size < 1024:
            raise RuntimeError(
                f"{out_path} is {size} bytes -- the codec accepted the open but wrote nothing. "
                "Try --codec MJPG."
            )

    if sidecar:
        write_sidecar(out_path, spec, rows, argv or [])

    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    n_absent = sum(1 for r in rows if r["absent"])
    print(
        f"{out_path.name}: {spec.frames} frames, "
        f"r {radii.min():.1f}-{radii.max():.1f} px, "
        f"step mean {step.mean():.1f} max {step.max():.1f} px"
        + (f", absent {n_absent}" if n_absent else "")
        + (f", max scale ratio {worst:.4f}" if spec.varies_scale else "")
    )
    return out_path


def _apply_overrides(spec: ClipSpec, args: argparse.Namespace) -> ClipSpec:
    if args.frames is not None:
        spec.frames = args.frames
    if args.fps is not None:
        spec.fps = args.fps
    if args.size is not None:
        w, _, h = args.size.partition("x")
        spec.width, spec.height = int(w), int(h)
    if args.seed is not None:
        spec.seed = args.seed
    if args.speed is not None:
        spec.speed_max = args.speed
    if args.radius is not None:
        spec.radius = args.radius
    if args.radius_min is not None:
        spec.radius_min = args.radius_min
    if args.radius_max is not None:
        spec.radius_max = args.radius_max
    if args.scale_period is not None:
        spec.scale_period = args.scale_period
    if args.occluder_value is not None:
        spec.occluder_value = args.occluder_value
    return spec


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true", help="generate the whole clip suite")
    parser.add_argument("--clip", choices=sorted(PRESETS), help="generate one preset")
    parser.add_argument("--out", type=Path, help="output path (single clip)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--size", help="WxH, e.g. 1280x720")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed", type=float, help="max speed, px/frame")
    parser.add_argument("--radius", type=float, help="fixed radius (non-scale clips)")
    parser.add_argument("--radius-min", type=float)
    parser.add_argument("--radius-max", type=float)
    parser.add_argument("--scale-period", type=float, help="frames per full scale cycle")
    parser.add_argument("--occluder-value", type=int, help=f"grey level of bars (default {OCCLUDER_VALUE})")
    parser.add_argument("--codec", help="fourcc, e.g. mp4v / MJPG / XVID")
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="sidecar and stats only, no video")
    parser.add_argument("--force", action="store_true", help="emit even if the scale rate is too fast")
    args = parser.parse_args(argv)

    if not args.all and not args.clip:
        parser.error("pass --all or --clip <name>")

    names = sorted(PRESETS) if args.all else [args.clip]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = _apply_overrides(PRESETS[name], args)
        out = args.out if (args.out and not args.all) else args.out_dir / f"{spec.name}.mp4"
        generate(
            spec,
            out,
            codec=args.codec,
            sidecar=not args.no_sidecar,
            preview=args.preview,
            dry_run=args.dry_run,
            force=args.force,
            argv=argv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
