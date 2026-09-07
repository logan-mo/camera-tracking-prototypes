"""Shared discriminative-correlation-filter maths.

THE CONVENTION -- read this before touching any tracker
=======================================================

Filters are stored **already conjugated** (Bolme's ``H*``), and the response is

    r = irfft2( sum_c  Hstar_c * F(z_c) )

Training accordingly puts the conjugate on the *patch*:

    A_c = F(y) * conj(F(x_c))          numerator, per channel
    B   = sum_c F(x_c) * conj(F(x_c))  denominator, shared across channels
    Hstar = A / (B + lambda)

Putting ``conj`` on the wrong operand is the single most likely bug in this
codebase.  It mirrors the response map, so the peak appears at ``(-dy, -dx)``
and the tracker moves *away* from the target at exactly twice the true speed.
:func:`self_test` catches it in one call and is run by ``__main__``.

Gaussian labels are constructed **already in wrapped order** -- peak at raw index
``(0, 0)`` -- by :func:`gaussian_label_2d`.  Consequently **``fftshift`` is never
called anywhere in the tracker path**.  It appears exactly once in this package,
in :mod:`trackers.debugview`, so a human can look at a centred response map.  Any
``fftshift`` you find in tracker code is a bug.

Everything is ``float64`` and channel-first ``(C, h, w)``.  Measured on this
build: ``rfft2`` on float64 takes 2.46 ms for (31, 64, 64) against 6.82 ms for
``fft2`` and 5.48 ms for float32 -- float32 is *slower* here, so the usual
"use float32 for speed" instinct is wrong on this machine.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

__all__ = [
    "apce",
    "dcf_detect",
    "dcf_train",
    "fft",
    "gaussian_label_1d",
    "gaussian_label_2d",
    "get_subwindow",
    "ifft",
    "next_fast_size",
    "peak_subpixel",
    "psr",
    "resize_patch",
    "self_test",
    "subwindow_centre",
]


def fft(x: np.ndarray) -> np.ndarray:
    """Real FFT over the last two axes. ``(..., h, w) -> (..., h, w//2+1)``."""
    return np.fft.rfft2(np.asarray(x, dtype=np.float64), axes=(-2, -1))


def ifft(X: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`fft`.

    ``shape`` is mandatory: ``rfft2`` discards the parity of the last axis, so
    without it a 63-wide patch comes back 64 wide.
    """
    return np.fft.irfft2(X, s=shape, axes=(-2, -1))


def next_fast_size(n: int) -> int:
    """Next 2/3/5-smooth size. ``cv.getOptimalDFTSize`` doubles as numpy's
    ``next_fast_len`` -- pocketfft is fast on exactly these sizes too."""
    return int(cv.getOptimalDFTSize(int(n)))


def _wrapped_coords(n: int) -> np.ndarray:
    """0, 1, ..., n//2, -(n//2-1), ..., -1 -- i.e. fftshifted order, no shift call."""
    return (np.arange(n) + n // 2) % n - n // 2


def gaussian_label_2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Gaussian regression target with its peak at raw index (0, 0)."""
    h, w = shape
    iy = _wrapped_coords(h).astype(np.float64)
    ix = _wrapped_coords(w).astype(np.float64)
    return np.exp(-0.5 * (iy[:, None] ** 2 + ix[None, :] ** 2) / (sigma * sigma))


def gaussian_label_1d(n: int, sigma: float) -> np.ndarray:
    """1-D Gaussian target with its peak at raw index 0 (for the scale filter)."""
    i = _wrapped_coords(n).astype(np.float64)
    return np.exp(-0.5 * (i * i) / (sigma * sigma))


def get_subwindow(img: np.ndarray, centre: tuple[float, float],
                  size: tuple[int, int]) -> np.ndarray:
    """Extract a window, replicate-padding at the frame border.

    Built from clamped index arrays rather than ``cv.getRectSubPix``: the target
    reaches the frame edges in this content, and ``getRectSubPix`` errors on a
    ROI far outside the image, so this path is exercised rather than
    hypothetical.  Four lines, no padded copy, handles a window entirely
    off-frame.
    """
    out_h, out_w = int(size[0]), int(size[1])
    cx, cy = float(centre[0]), float(centre[1])
    y0 = int(round(cy - out_h / 2.0))
    x0 = int(round(cx - out_w / 2.0))
    ys = np.clip(np.arange(out_h) + y0, 0, img.shape[0] - 1)
    xs = np.clip(np.arange(out_w) + x0, 0, img.shape[1] - 1)
    return img[np.ix_(ys, xs)] if img.ndim == 2 else img[np.ix_(ys, xs)][..., :]


def subwindow_centre(centre: tuple[float, float], size: tuple[int, int]) -> tuple[float, float]:
    """The centre :func:`get_subwindow` actually used, after integer rounding.

    Extraction snaps its origin to whole pixels, so the patch centre differs
    from the requested centre by up to half a pixel.  A response peak is
    measured relative to the *patch*, so adding it to the requested centre
    instead of the effective one injects that residual as error on every frame.

    At a few tenths of a pixel this is invisible in centre error -- but DSST
    feeds its centre straight into the scale filter, where an off-centre patch
    reads as a *different scale*, and the resulting scale error shrinks the
    search window and feeds back.  That loop is what makes a half-pixel bug
    matter.
    """
    out_h, out_w = int(size[0]), int(size[1])
    ey = float(int(round(centre[1] - out_h / 2.0))) + out_h / 2.0
    ex = float(int(round(centre[0] - out_w / 2.0))) + out_w / 2.0
    return ex, ey


def resize_patch(patch: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize, choosing interpolation by direction.

    ``INTER_LINEAR`` aliases when downscaling.  DSST downscales roughly half of
    its 33 scale patches and upscales the other half, so a single fixed
    interpolation puts a systematic asymmetry into the scale response and the
    filter acquires a preferred direction of scale change.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    interp = cv.INTER_AREA if (out_h < patch.shape[0] or out_w < patch.shape[1]) else cv.INTER_LINEAR
    return cv.resize(patch, (out_w, out_h), interpolation=interp)


def dcf_train(x_hat: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numerator/denominator for a linear multi-channel filter.

    ``x_hat`` is ``(C, h, w')`` and ``y_hat`` is ``(h, w')``.  MOSSE calls this
    with ``C=1``; DSST's translation filter calls it with ``C`` channels.  KCF
    (kernelised) and CSRT (ADMM) deliberately do not use it.
    """
    a = y_hat[None, ...] * np.conj(x_hat)
    b = np.sum(x_hat * np.conj(x_hat), axis=0).real
    return a, b


def dcf_detect(a: np.ndarray, b: np.ndarray, z_hat: np.ndarray,
               lam: float, shape: tuple[int, int]) -> np.ndarray:
    """Correlation response for a filter stored as ``(a, b)``."""
    num = np.sum(a * z_hat, axis=0)
    return ifft(num / (b + lam), shape)


def peak_subpixel(r: np.ndarray) -> tuple[float, float, float]:
    """Locate the response peak. Returns ``(dy, dx, peak_value)`` as signed shifts.

    Operates on the **raw, unshifted** map.  Two details that are easy to get
    wrong and produce very recognisable symptoms:

    * The parabolic fit uses neighbours **with modulo wraparound**.
    * The wrapped index is converted to a signed shift.  Forgetting that makes
      the tracker jump by exactly the window size the first time the target
      moves in the negative direction.
    """
    h, w = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r[iy, ix])

    def refine(prev: float, cur: float, nxt: float) -> float:
        denom = prev - 2.0 * cur + nxt
        if abs(denom) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (prev - nxt) / denom, -0.5, 0.5))

    dy = refine(float(r[(iy - 1) % h, ix]), peak, float(r[(iy + 1) % h, ix]))
    dx = refine(float(r[iy, (ix - 1) % w]), peak, float(r[iy, (ix + 1) % w]))

    sy = iy - h if iy > h // 2 else iy
    sx = ix - w if ix > w // 2 else ix
    return float(sy) + dy, float(sx) + dx, peak


def psr(r: np.ndarray, exclude: int = 11) -> float:
    """Peak-to-sidelobe ratio, with the peak window masked (wraparound-aware).

    Faithful to Bolme, but note that on this content the background has
    **exactly zero variance**, so the sidelobe std is tiny and PSR reads far
    above Bolme's 20-60 range.  A hard ``PSR < 7 -> lost`` rule would never fire,
    not even during full occlusion.  Trackers here therefore gate on confidence
    relative to a running median (see :class:`trackers.base.BaseTracker`) and
    report raw PSR only as a diagnostic.
    """
    h, w = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r[iy, ix])

    half = max(1, int(exclude) // 2)
    mask = np.ones((h, w), dtype=bool)
    ys = (np.arange(iy - half, iy + half + 1)) % h
    xs = (np.arange(ix - half, ix + half + 1)) % w
    mask[np.ix_(ys, xs)] = False

    side = r[mask]
    if side.size == 0:
        return 0.0
    return float((peak - side.mean()) / (side.std() + 1e-9))


def apce(r: np.ndarray) -> float:
    """Average peak-to-correlation energy.

    Normalised by the whole map's energy rather than by a near-zero sidelobe
    std, so unlike PSR it stays meaningful on a flat background.
    """
    peak = float(r.max())
    low = float(r.min())
    denom = float(np.mean((r - low) ** 2))
    return float((peak - low) ** 2 / (denom + 1e-9))


def self_test(verbose: bool = True) -> bool:
    """Assert the conventions hold. Run this before trusting any tracker.

    Trains a filter on ``x`` with label ``y``, then correlates it back against
    ``x``.  The peak must land at raw index ``(0, 0)``.  If the conjugate is on
    the wrong operand this fails immediately.
    """
    rng = np.random.default_rng(0)
    shape = (64, 64)
    x = rng.standard_normal((1, *shape))
    y = gaussian_label_2d(shape, 2.0)

    a, b = dcf_train(fft(x), fft(y))
    r = dcf_detect(a, b, fft(x), 1e-6, shape)

    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    ok_peak = (iy, ix) == (0, 0)

    # A known shift must be recovered with the correct sign.
    shift_y, shift_x = 7, -5
    z = np.roll(x, (shift_y, shift_x), axis=(-2, -1))
    rz = dcf_detect(a, b, fft(z), 1e-6, shape)
    dy, dx, _ = peak_subpixel(rz)
    ok_shift = abs(dy - shift_y) < 0.5 and abs(dx - shift_x) < 0.5

    # The label itself must peak at the origin.
    ok_label = int(np.argmax(y)) == 0

    if verbose:
        print(f"  peak at raw (0,0):        {'PASS' if ok_peak else 'FAIL'}  got ({iy}, {ix})")
        print(f"  recovers shift ({shift_y:+d},{shift_x:+d}):  "
              f"{'PASS' if ok_shift else 'FAIL'}  got ({dy:+.2f}, {dx:+.2f})")
        print(f"  label peaks at index 0:   {'PASS' if ok_label else 'FAIL'}")
    return bool(ok_peak and ok_shift and ok_label)


if __name__ == "__main__":
    print("correlation.py convention self-test:")
    raise SystemExit(0 if self_test() else 1)
