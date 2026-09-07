"""Run every tracker over every clip and emit the comparison table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import registry
from .metrics import (
    HEADLINE_COLUMNS,
    RunStats,
    ascii_series_plot,
    ascii_success_plot,
    format_table,
    write_frames_csv,
    write_summary_csv,
)
from .runner import RunConfig, run

__all__ = ["main"]

DEFAULT_CLIPS = [
    "data/synth_baseline.mp4",
    "data/synth_scale.mp4",
    "data/synth_occlusion.mp4",
    "data/synth_fast.mp4",
    "tracking_test_video.mp4",
]


def _ensure_clips(paths: list[Path], out_dir: Path) -> list[Path]:
    """Generate any missing synthetic clip rather than silently skipping it."""
    missing = [p for p in paths if not p.exists() and p.name.startswith("synth_")]
    if missing:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.make_test_videos import main as gen_main

        print(f"generating {len(missing)} missing clip(s) into {out_dir}...")
        gen_main(["--all", "--out-dir", str(out_dir)])
    return [p for p in paths if p.exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark every tracker over every clip.")
    parser.add_argument("--videos", nargs="*", type=Path)
    parser.add_argument("--trackers", default="all",
                        help="'all' or a comma-separated subset")
    parser.add_argument("--detector", default="blob")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat each run and keep the median FPS (timing stability)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-auto-generate", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--plots", action="store_true", help="write ASCII success/scale plots")
    args = parser.parse_args(argv)

    videos = args.videos or [Path(p) for p in DEFAULT_CLIPS]
    if not args.no_auto_generate:
        videos = _ensure_clips(videos, args.out.parent / "data" if args.out.name == "data" else Path("data"))
    videos = [v for v in videos if v.exists()]
    if not videos:
        print("no videos to benchmark", file=sys.stderr)
        return 1

    names = (
        [n for n in registry.DEFAULT_ORDER if n in registry.TRACKER_SPECS]
        if args.trackers == "all"
        else [n.strip().lower() for n in args.trackers.split(",") if n.strip()]
    )

    all_stats: list[RunStats] = []
    per_clip: dict[str, list[RunStats]] = {}
    records_by_key: dict[tuple[str, str], list] = {}
    args.out.mkdir(parents=True, exist_ok=True)

    for video in videos:
        clip_stats: list[RunStats] = []
        for name in names:
            attempts = []
            best_records = None
            for _ in range(max(1, args.repeat)):
                cfg = RunConfig(
                    video=video, tracker=name, detector=args.detector,
                    show=False, max_frames=args.max_frames, draw_trail=False,
                )
                try:
                    result = run(cfg)
                except ImportError:
                    print(f"  {name:<9} {video.name:<24} SKIP (not implemented)")
                    attempts = []
                    break
                except Exception as exc:  # noqa: BLE001 - one bad tracker must not kill the matrix
                    print(f"  {name:<9} {video.name:<24} ERROR: {exc}", file=sys.stderr)
                    if args.fail_fast:
                        raise
                    attempts = []
                    break
                attempts.append(result.stats)
                best_records = result.records
            if not attempts:
                continue
            # Median FPS across repeats; the first run pays page-cache and
            # interpreter warm-up costs.
            stats = attempts[0]
            if len(attempts) > 1:
                stats.fps_median = float(np.median([a.fps_median for a in attempts]))
                stats.update_ms_median = float(np.median([a.update_ms_median for a in attempts]))
            clip_stats.append(stats)
            all_stats.append(stats)
            if best_records is not None:
                records_by_key[(name, video.stem)] = best_records
                write_frames_csv(best_records, args.out / f"frames_{name}_{video.stem}.csv")
            print(f"  {name:<9} {video.name:<24} IoU {stats.mean_iou:.3f}  "
                  f"ScaleErr {stats.scale_err:.3f}  {stats.fps_median:.0f} fps")
        per_clip[video.stem] = clip_stats

    print()
    for clip, stats in per_clip.items():
        if not stats:
            continue
        print(f"=== {clip} ===")
        print(format_table(stats))
        print()

    if all_stats:
        write_summary_csv(all_stats, args.out / "summary.csv")
        print(f"summary -> {args.out / 'summary.csv'}")

    if args.plots:
        for clip, stats in per_clip.items():
            if not stats:
                continue
            lines = [ascii_success_plot(stats)]
            series = []
            for name in ("kcf", "dsst", "mosse", "csrt"):
                rec = records_by_key.get((name, clip))
                if rec:
                    series.append((name, np.array([r.scale_ratio for r in rec])))
            if series:
                lines.append("")
                lines.append(
                    ascii_series_plot(series, label="scale_ratio vs frame (1.0 = perfect)")
                )
            (args.out / f"plots_{clip}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"plots -> {args.out}/plots_*.txt")

    scale_stats = per_clip.get("synth_scale", [])
    if scale_stats:
        print("=== headline: synth_scale (the scale-variation clip) ===")
        print(format_table(scale_stats, HEADLINE_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
