"""Feature backends for the correlation-filter trackers.

Two backends: raw intensity and a from-scratch 31-channel Felzenszwalb fHOG.

Which to use, and why both exist
--------------------------------
Intensity is the default fast path.  On this content the target is a filled
36 px disc of value 255 on a background of value 15 with *zero* variance -- about
as high-SNR as a correlation template gets.  fHOG reduces that to gradient
energy on a ~1 px boundary ring, discarding the ~1000 px of interior signal that
make the peak sharp, and at ``cell_size=4`` it quantises the response map to 4 px
before subpixel refinement.  On a 36 px target that is a pure accuracy loss for
31x the compute.

fHOG is built anyway, and is **mandatory for CSRT**, because CSR-DCF's
channel-reliability weights are identically vacuous with one channel.  Shipping
CSRT on intensity would silently stub out one of the paper's three named
contributions while still calling the result CSR-DCF.  It also matters for the
eventual gimbal-camera target, where a textured background would destroy an
intensity-only tracker.

Verifying fHOG with no reference implementation
-----------------------------------------------
There is no scipy/dlib here to diff against, so :func:`fhog_self_test` checks
two structural properties instead:

* **Contrast invariance** -- scaling the patch by 2.0 must leave channels 0-26
  essentially unchanged.  That is precisely the contract of block normalisation.
* **Orientation binning** -- a pure vertical edge must put all folded energy in
  insensitive bin 0.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

__all__ = [
    "FeatureExtractor",
    "fhog",
    "fhog_self_test",
    "hann2d",
    "preprocess_gray",
    "preprocess_mosse",
]

_EPS = 1e-4
_TRUNC = 0.2
_NORM4 = 0.2357022603955158  # 1/sqrt(18)


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    """Periodic 2-D Hann window.

    ``np.hanning(n)`` is exactly zero at both endpoints, annihilating the outer
    ring of every patch; the periodic form ``np.hanning(n+2)[1:-1]`` is not.
    Use one form everywhere -- mixing them changes the effective padding.

    The window is not cosmetic for KCF: the circulant model treats every cyclic
    shift of the patch as a training sample, so without it the wraparound
    discontinuity at the patch edge becomes a real training sample and KCF
    produces a strong spurious response along the boundary.
    """
    h, w = shape
    return np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1])


def preprocess_gray(patch: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    """Grayscale -> float64, zero-mean, Hann-windowed. Returns ``(1, h, w)``."""
    x = np.asarray(patch, dtype=np.float64) / 255.0
    x = x - x.mean()
    if window is not None:
        x = x * window
    return x[None, ...]


def preprocess_mosse(patch: np.ndarray, window: np.ndarray | None = None,
                     use_log: bool = True) -> np.ndarray:
    """Bolme's recipe: log(1+x), zero mean, unit norm, Hann.

    ``use_log`` is a flag rather than a constant on purpose.  The log is a
    contrast-normalisation step designed for real footage; on this content it
    compresses the target/background ratio from 17x (255/15) to 2x
    (5.55/2.77), so it should actively *hurt*.  Both are benchmarked.
    """
    x = np.asarray(patch, dtype=np.float64)
    if use_log:
        x = np.log(x + 1.0)
    x = x - x.mean()
    x = x / (np.linalg.norm(x) + 1e-9)
    if window is not None:
        x = x * window
    return x[None, ...]


def _gradients(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude and orientation, picking the strongest colour channel per pixel.

    ``ksize=1`` gives the canonical central difference [-1, 0, 1] that the
    reference uses, not a 3x3 Sobel kernel.
    """
    img = np.asarray(patch, dtype=np.float64)
    if img.ndim == 2:
        gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
        gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=1)
    else:
        gxs = np.stack([cv.Sobel(img[..., c], cv.CV_64F, 1, 0, ksize=1) for c in range(img.shape[2])])
        gys = np.stack([cv.Sobel(img[..., c], cv.CV_64F, 0, 1, ksize=1) for c in range(img.shape[2])])
        mags = gxs * gxs + gys * gys
        pick = np.argmax(mags, axis=0)[None, ...]
        gx = np.take_along_axis(gxs, pick, axis=0)[0]
        gy = np.take_along_axis(gys, pick, axis=0)[0]
    mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=False)
    return mag, ang


def fhog(patch: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """31-channel Felzenszwalb HOG. Returns ``(31, nc_y - 2, nc_x - 2)``.

    The outer cell ring is dropped, as tracking implementations do (rather than
    zeroed, as voc-release5 does), so **size the search window such that the
    post-crop map is the size you intended** -- this is a guaranteed off-by-one
    otherwise.
    """
    k = int(cell_size)
    mag, ang = _gradients(patch)
    h, w = mag.shape
    nc_y, nc_x = h // k, w // k
    if nc_y < 3 or nc_x < 3:
        raise ValueError(f"patch {h}x{w} too small for cell_size {k}")

    # -- orientation binning: hard assignment to 18 contrast-sensitive bins ---
    bins = np.floor(18.0 * ang / (2.0 * np.pi)).astype(np.int64) % 18

    # -- cell aggregation with bilinear SPATIAL interpolation ----------------
    # Spatial interpolation matters: without it the descriptor jumps
    # discontinuously under 1 px target shifts, which is the regime a tracker
    # lives in.  Orientation assignment stays hard, as in the reference.
    yy, xx = np.mgrid[0:h, 0:w]
    fy = (yy - (k - 1) / 2.0) / k
    fx = (xx - (k - 1) / 2.0) / k
    cy0 = np.floor(fy).astype(np.int64)
    cx0 = np.floor(fx).astype(np.int64)
    wy1 = fy - cy0
    wx1 = fx - cx0

    pad_y, pad_x = nc_y + 2, nc_x + 2
    acc = np.zeros(pad_y * pad_x * 18, dtype=np.float64)
    # Four bincount calls, one per bilinear corner.  np.add.at takes an
    # unbuffered slow path and is roughly an order of magnitude slower here.
    for dy, wy in ((0, 1.0 - wy1), (1, wy1)):
        for dx, wx in ((0, 1.0 - wx1), (1, wx1)):
            iy = np.clip(cy0 + dy + 1, 0, pad_y - 1)
            ix = np.clip(cx0 + dx + 1, 0, pad_x - 1)
            flat = (iy * pad_x + ix) * 18 + bins
            acc += np.bincount(flat.ravel(), weights=(mag * wy * wx).ravel(),
                               minlength=acc.size)
    cells = acc.reshape(pad_y, pad_x, 18)[1:-1, 1:-1, :]

    # -- energy and the four 2x2 normalisers ---------------------------------
    insensitive = cells[..., :9] + cells[..., 9:]
    energy = np.sum(insensitive * insensitive, axis=-1)
    ep = np.pad(energy, 1, mode="edge")
    block = ep[:-1, :-1] + ep[1:, :-1] + ep[:-1, 1:] + ep[1:, 1:]
    norms = [block[1:, 1:], block[1:, :-1], block[:-1, 1:], block[:-1, :-1]]
    norms = [1.0 / np.sqrt(n + _EPS) for n in norms]

    # -- 31-channel assembly --------------------------------------------------
    out = np.zeros((nc_y, nc_x, 31), dtype=np.float64)
    for j, n in enumerate(norms):
        n3 = n[..., None]
        t_ins = np.minimum(_TRUNC, insensitive * n3)
        t_sen = np.minimum(_TRUNC, cells * n3)
        out[..., 0:9] += 0.5 * t_ins
        out[..., 9:27] += 0.5 * t_sen
        out[..., 27 + j] = _NORM4 * t_sen.sum(axis=-1)

    return np.ascontiguousarray(out[1:-1, 1:-1, :].transpose(2, 0, 1))


class FeatureExtractor:
    """Selects a feature backend and applies the Hann window consistently."""

    def __init__(self, kind: str = "gray", cell_size: int = 4, use_log: bool = False) -> None:
        if kind not in {"gray", "fhog", "fhog+gray", "mosse"}:
            raise ValueError(f"unknown feature kind {kind!r}")
        self.kind = kind
        self.cell_size = int(cell_size) if kind != "gray" else 1
        self.use_log = use_log
        self._window: np.ndarray | None = None
        self._window_shape: tuple[int, int] | None = None

    def _hann(self, shape: tuple[int, int]) -> np.ndarray:
        if self._window_shape != shape:
            self._window = hann2d(shape)
            self._window_shape = shape
        return self._window  # type: ignore[return-value]

    def out_shape(self, patch_hw: tuple[int, int]) -> tuple[int, int]:
        if self.kind == "gray" or self.kind == "mosse":
            return patch_hw
        k = self.cell_size
        return (patch_hw[0] // k - 2, patch_hw[1] // k - 2)

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        if self.kind in {"gray", "mosse"}:
            gray = patch if patch.ndim == 2 else cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
            window = self._hann(gray.shape[:2])
            if self.kind == "mosse":
                return preprocess_mosse(gray, window, use_log=self.use_log)
            return preprocess_gray(gray, window)

        feats = fhog(patch, self.cell_size)
        if self.kind == "fhog+gray":
            gray = patch if patch.ndim == 2 else cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
            small = cv.resize(
                np.asarray(gray, dtype=np.float64) / 255.0,
                (feats.shape[2], feats.shape[1]),
                interpolation=cv.INTER_AREA,
            )
            feats = np.concatenate([feats, (small - small.mean())[None, ...]], axis=0)

        window = self._hann(feats.shape[1:])
        return feats * window[None, ...]


def fhog_self_test(verbose: bool = True) -> bool:
    """Structural checks for fHOG, since there is no reference to diff against."""
    rng = np.random.default_rng(0)
    patch = (rng.random((64, 64)) * 255).astype(np.float64)

    a = fhog(patch, 4)
    b = fhog(patch * 2.0, 4)
    # Block normalisation makes channels 0-26 invariant to a contrast scale.
    denom = np.abs(a[:27]).max() + 1e-12
    contrast_err = float(np.abs(a[:27] - b[:27]).max() / denom)
    ok_contrast = contrast_err < 1e-6

    # A pure vertical edge: gradient points along x, so folded energy lands in
    # insensitive bin 0.
    edge = np.zeros((64, 64), dtype=np.float64)
    edge[:, 32:] = 255.0
    fe = fhog(edge, 4)
    ins = fe[:9]
    per_bin = ins.reshape(9, -1).sum(axis=1)
    ok_edge = int(np.argmax(per_bin)) == 0

    ok_shape = a.shape == (31, 14, 14)
    ok_finite = bool(np.all(np.isfinite(a)))

    if verbose:
        print(f"  shape (31,14,14) for 64x64/k=4: {'PASS' if ok_shape else 'FAIL'}  got {a.shape}")
        print(f"  contrast invariance ch0-26:     {'PASS' if ok_contrast else 'FAIL'}  "
              f"max rel err {contrast_err:.2e}")
        print(f"  vertical edge -> ins. bin 0:    {'PASS' if ok_edge else 'FAIL'}  "
              f"argmax {int(np.argmax(per_bin))}")
        print(f"  all finite:                     {'PASS' if ok_finite else 'FAIL'}")
    return bool(ok_contrast and ok_edge and ok_shape and ok_finite)


if __name__ == "__main__":
    print("features.py fHOG self-test:")
    raise SystemExit(0 if fhog_self_test() else 1)
