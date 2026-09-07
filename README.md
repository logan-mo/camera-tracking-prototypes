# camera-tracking-prototypes

Six single-object trackers behind one interface, plus a benchmark that puts real
numbers behind the **scale-variation failure mode**: a fixed-size-box tracker
loses a target that grows in frame as a drone closes distance, while a tracker
that estimates size explicitly does not.

```bash
uv run tools/make_test_videos.py --all      # generate the test clips
uv run benchmark.py                         # run every tracker over every clip
uv run track.py data/synth_scale.mp4 -t dsst   # watch one tracker
```

## Two things you need to know first

**1. OpenCV 5.0 removed every classical tracker.** This project is on
`opencv-python 5.0.0.93`, where `cv2.legacy`, `TrackerKCF`, `TrackerCSRT` and
`TrackerMOSSE` no longer exist — only `TrackerMIL`/`Vit`/`Nano`/`DaSiamRPN`
remain. Tutorials that call `cv2.legacy.TrackerMOSSE_create()` predate this.
So **MOSSE, KCF, DSST and CSRT here are from-scratch NumPy implementations**,
not OpenCV wrappers. (DSST was never in mainline OpenCV in any version.) The
only dependencies are numpy and opencv — no scipy, torch, ultralytics, filterpy
or lap.

**2. `tracking_test_video.mp4` cannot answer the scale question.** Measured
across all 2835 frames, its circle is **34–36 px wide on every single frame** —
there is no scale variation to fail on. Worse, its target moves a **median of
24.6 px/frame (0.7× its own diameter), peaking at 140 px**. A correlation filter
can only follow about half a search window per frame, and sweeping padding on a
from-scratch MOSSE shows no setting rescues it:

| padding | window | mean err | within 20 px |
| --- | --- | --- | --- |
| 2.0 (standard) | 72² | 559 px | 8.4% |
| 4.0 | 144² | 197 px | **56.5%** |
| 9.0 | 324² | 190 px | 5.4% |

That is a property of the clip, not of any implementation. It is kept in the
bench as a stress case, and `data/synth_fast.mp4` reproduces its motion regime
(mean 28.3 vs 28.5 px/frame) so the effect can be quantified rather than
hand-waved.

## The clips

`tools/make_test_videos.py` generates five clips with exact ground truth in a
`<stem>.gt.json` sidecar (sub-pixel centre, radius, and a `visible` fraction —
the only correct ground truth for occluded frames, since thresholding cannot
recover a box that is hidden).

| clip | purpose |
| --- | --- |
| `synth_baseline` | sanity: everything should pass |
| `synth_scale` | **the money clip** — radius sweeps 12↔48 px (4× linear, 16× area) |
| `synth_occlusion` | occlusion and re-acquisition |
| `synth_fast` | reproduces the source clip's motion regime |
| `synth_scale_occlusion` | combined stress |

Speed is capped at ~5 px/frame on the scale clip *on purpose*: translation then
costs nothing, so the benchmark measures **scale handling rather than
search-window size**. The scale period is constrained so the per-frame radius
ratio (1.0122) stays inside DSST's 1.02 scale pool — otherwise the clip, not the
tracker, is what's being measured, and the generator refuses to emit a clip that
violates this without `--force`.

## Reading the results table

```
Tracker  Type  Clip  Frames  IoU  AUC  Prec@20  Scale×  ScaleErr  F2F  Fail%  FPS  ms/f
```

- **`ScaleErr`** is `mean |log₂(pred/gt)|` in octaves — symmetric and
  non-cancelling. This is the column that separates the scale-adaptive trackers
  from the fixed-box ones. A plain mean scale ratio is *actively misleading* on a
  sinusoidal scale clip, because over- and under-estimates cancel to ≈1.0 for a
  tracker that is wildly wrong in both directions.
- **`IoU` and `Prec@20` are reported separately, never merged.** A disc is
  centrally symmetric, so a fixed-box tracker's correlation peak stays
  well-centred as the target grows — its centre error looks *fine* and only the
  box is wrong. **The scale failure is invisible in centre error and shows up
  only in IoU and ScaleErr.** High `Prec@20` together with high `ScaleErr` is the
  signature of pure scale failure rather than tracking loss.
- **`Type`** matters: `SORT` is `detection`, not `appearance`. Its box is
  re-derived from the detector every frame, so it needs no scale handling and
  scores near-zero scale error. That is *not* SORT beating DSST at scale
  estimation — it is measuring the detector, and it is unavailable the moment you
  have no detector for your target.
- **`FPS`** times `tracker.update()` only — no frame reads, colour conversion,
  drawing, `imshow`/`waitKey` or metric computation. Frame 0 is excluded and
  reported separately, since correlation filters do heavy FFT setup on init.

## Results

### `synth_scale` — the scale-variation clip (the question you actually asked)

| Tracker | Type | IoU | Prec@20 | Scale× | **ScaleErr** | Fails at | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle | reference | 1.000 | 1.000 | 1.000 | 0.000 | never | — |
| SORT | detection | 0.953 | 1.000 | 1.006 | 0.027 | never | 441 |
| **DSST** | appearance | **0.894** | 1.000 | 0.971 | **0.084** | never | 46 |
| **CSRT** | appearance | **0.848** | 1.000 | 0.965 | **0.122** | never | 6 |
| *FixedBox* | *reference* | *0.455* | *1.000* | *0.836* | *0.637* | *f61* | — |
| KCF | appearance | 0.405 | 0.743 | 0.836 | **0.637** | f61 | 161 |
| MOSSE | appearance | 0.387 | 0.624 | 0.836 | **0.637** | f61 | 154 |
| KLT | flow | 0.291 | 0.611 | 0.836 | 0.630 | f61 | 134 |
| Static | reference | 0.036 | 0.053 | 0.836 | 0.637 | f53 | — |

**KCF and MOSSE score ScaleErr 0.637 — identical to `FixedBox`, and identical to
the analytic prediction (2/π = 0.6366).** They are not broken; they are behaving
exactly as fixed-box trackers must. DSST cuts that to **0.084, a 7.6× reduction**,
and more than doubles IoU (0.405 → 0.894). That is the whole argument, measured.

DSST is also **3.5× faster than CSRT here** (46 vs 6 fps) for *better* accuracy on
this content — consistent with DSST being the lighter fix.

### `synth_occlusion` — where CSRT earns its cost

| Tracker | IoU | Prec@20 | Fail% | Fails at | FPS |
| --- | --- | --- | --- | --- | --- |
| DSST | 0.958 | 0.984 | 1.6% | f616 | 66 |
| **CSRT** | **0.922** | **0.999** | **0.2%** | **never** | 14 |
| SORT | 0.911 | 0.988 | 1.6% | f598 | 524 |
| KLT | 0.677 | 0.694 | 30.6% | f616 | 148 |
| MOSSE | 0.642 | 0.694 | 30.6% | f616 | 172 |
| KCF | 0.639 | 0.694 | 30.6% | f616 | 320 |

**CSRT is the only tracker that never fails**, at a 30× throughput cost against
KCF. Its spatial reliability map means it trains only on target pixels, so it
does not absorb the occluder. MOSSE, KCF and KLT all lose the target at the same
frame with the same 30.6% failure rate.

### `synth_fast` and `tracking_test_video` — everything fails

| Tracker | IoU on `synth_fast` | IoU on `tracking_test_video` |
| --- | --- | --- |
| DSST | 0.583 | 0.161 |
| KCF | 0.468 | 0.139 |
| CSRT | 0.370 | 0.299 |
| KLT | 0.317 | 0.120 |
| SORT | 0.278 | 0.298 |
| MOSSE | 0.275 | 0.086 |
| *FixedBox* | *1.000* | ***0.967*** |

**`FixedBox` — a perfect centre with a permanently frozen box — scores 0.967 on
`tracking_test_video`.** So that clip contains essentially no scale variation to
fail on, and every tracker's collapse there is **100% a translation failure**. The
per-frame motion (median 24.6 px against a 35 px target) simply exceeds what a
correlation filter can follow. On the real clip DSST's and CSRT's scale estimates
also blow up (Scale× 5.19 and 0.071) — once the target is lost, the scale filter
has nothing to lock onto and runs away, which is expected rather than a separate
defect.

Full tables in `results/summary.csv`; per-frame traces in
`results/frames_<tracker>_<clip>.csv`.

## Oracles bound the table

`Oracle` (returns ground truth) and `FixedBox` (perfect centre, frozen box size)
are in the table on purpose. They make a surprising number attributable to a
tracker rather than to the bench:

- `Oracle` **must** score IoU 1.000 / AUC 1.000 / ScaleErr 0.000. It does. Any
  deviation is a metrics bug.
- `FixedBox` **must** score ScaleErr 0.637 on `synth_scale` — the analytic value
  `mean|log₂| = (2/π)` for a box held at mid-scale while the target sweeps a
  sine. It measures **0.637**, and its IoU of 0.455 matches the analytic 0.4553.

So MOSSE and KCF landing on ScaleErr ≈ 0.637 is the **expected, correct** result,
not a bug — and if DSST failed to beat `FixedBox`, its scale filter would be
broken, which is knowable without arguing about tracker quality.

## Verification built into the code

Each of these is a `python -m` entry point and each caught a real bug during
development:

```bash
uv run python -m trackers.correlation                    # FFT/conjugate convention
uv run python -m trackers.features                       # fHOG structural self-test
uv run python -m trackers.groundtruth tracking_test_video.mp4 --report
```

- **`correlation.py`** — trains a filter on `x`, correlates it back against `x`,
  and asserts the peak lands at raw index `(0, 0)`, then that a known `(+7, −5)`
  shift is recovered with the right sign. Putting the conjugate on the wrong
  operand mirrors the response map and makes a tracker run *away* from the target
  at twice the true speed.
- **`features.py`** — with no reference implementation to diff against, fHOG is
  checked structurally: scaling the patch by 2.0 must leave channels 0–26
  unchanged (the exact contract of block normalisation; it holds to 2e−11), and a
  pure vertical edge must land in insensitive bin 0.
- **`groundtruth.py --report`** — must print `valid 2834 / 2835`. The one
  rejected frame is a garbage final decode. Note the ~70 frames a naive
  `area > 20` filter flags are **not** the circle clipped at a frame edge (a large
  blob touches a border on 0 of 2835 frames) — they contain a static arrow
  overlay with extent 0.014 against the disc's 0.72, which an extent predicate
  removes outright.
- **Generator ↔ extractor cross-check** — running threshold extraction against a
  clip's own sidecar validates both at once. This caught a **half-pixel bias**:
  OpenCV treats integer coordinates as pixel *centres*, while the bbox convention
  treats pixel `i` as covering `[i, i+1)`. It showed as a +0.50 px signed offset
  against 0.15 px of genuine antialiasing scatter.
- **`ScaleFilter.centre_fraction`** — the fraction of frames whose scale response
  peaked at "no change". **Above ~0.9 on a clip where the target demonstrably
  changes size, the scale filter is dead.** It raises no error and still reports
  plausible boxes, so nothing else would tell you.

## Honest caveats

- **CSRT here is CSR-DCF-*lite***: the paper's Markov-random-field regularisation
  of the spatial reliability map is replaced by morphological close/open plus
  largest-connected-component selection. That matches the de facto reference —
  OpenCV's own legacy `TrackerCSRT` also omits the full MRF — but it is a
  substitution and is labelled as one.
- The clips are effectively **monochrome**, so CSRT's colour model is a 1-D
  grayscale histogram, not the usual HSV. On a 255 disc over a zero-variance
  background, backprojection produces an essentially perfect mask, so **CSRT looks
  better here than it would on real footage**.
- **PSR is not a usable lost-target signal on this content.** The background has
  exactly zero variance, so Bolme's canonical `PSR < 7` rule would never fire —
  not even during full occlusion. Confidence is therefore relative to a running
  median of PSR, with APCE alongside; raw PSR is still reported so the
  content-dependence stays visible.
- **DSST and CSRT use a larger search padding (3.0) than canonical DSST (1.0).**
  This is structural, not a fudge: the scale clip grows the target 4× linearly, so
  a window of 2× the *init* box is exactly filled at maximum scale, leaving no
  context — the centre then drifts, the scale filter falls behind, the window
  shrinks with it, and the two feed each other. Sweeping only this parameter:
  `1.0 → ScaleErr 0.478, fails at frame 375`; `2.0 → 0.306, fails at 750`;
  `3.0 → 0.084, never fails`. That 0.084 matches what the scale filter achieves
  when fed ground-truth centres (0.087), i.e. translation stops being the
  bottleneck. Cost is ~5% of throughput.
- The four trackers use different paddings, and a larger window is *part of* why
  a tracker survives occlusion. For conclusions attributable to the algorithms
  rather than to window size, run an ablation with padding matched.
- **SORT's default `iou_threshold=0.3` breaks under fast motion** — not because
  detection fails, but because the association gate does: when per-frame
  displacement exceeds the target size, the predicted box and the new detection
  do not overlap, so the track is re-born every frame. On `synth_fast`, lowering
  it to 0.0 takes IoU from 0.278 to 0.576 and Prec@20 from 0.31 to 0.82. The
  reference default is kept; the knob is `--sort-max-age` / `iou_threshold`.

## Layout

```
track.py / benchmark.py      entry points (thin shims)
tools/make_test_videos.py    synthetic clip generator + ground-truth sidecars
trackers/
  types.py                   BBox + Detection (sole owner of the convention)
  base.py                    tracker interface + the shared update gate
  correlation.py             FFT convention, labels, peak, PSR/APCE, patches
  features.py                intensity + 31-channel fHOG
  scale.py                   1-D DSST scale filter (shared by DSST and CSRT)
  mosse.py kcf.py dsst.py csrt.py klt.py sort.py
  detectors.py               Detector protocol + BlobDetector (YOLO seam)
  groundtruth.py             sidecar or threshold extraction
  metrics.py                 IoU, ScaleErr, success/precision, timing, tables
  oracles.py                 Oracle / FixedBox / Static
  registry.py runner.py benchmark.py
sparse-optical-flow.py       OpenCV tutorial reference (productionised as klt.py)
dense-optical-flow.py        OpenCV tutorial reference (a motion field, not a tracker)
```

Adding a YOLO backend requires **no change to `sort.py`**: `sort.py` imports
`Detection` from `types.py`, never from `detectors.py`, and the CLI takes a
detector *spec string* (`--detector yolo:model.onnx`). `cv2.dnn` is present in
OpenCV 5.0, so an ONNX model needs no ultralytics or torch.
