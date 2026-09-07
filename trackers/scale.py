"""The 1-D scale-space correlation filter from DSST (Danelljan et al. 2014).

This is the piece that actually fixes the scale-variation failure.  It is a
completely separate filter from the translation filter, operating on a 1-D
"scale axis": at the current centre it extracts one patch per scale in a
geometric pool, resizes each to a fixed model size, flattens each to a feature
vector, and correlates across the scale dimension.

CSRT imports this class verbatim -- the CSR-DCF paper uses the DSST scale filter
unchanged -- so this is shared code rather than a reimplementation.

The trap that silently kills this filter
----------------------------------------
Standard DSST sets ``scale_model_max_area = 512``.  For a 36x36 target (area
1296) that downscales the scale model to **22x22**, where adjacent scales at
step 1.02 differ by ``0.02 * 22 = 0.44 px`` -- below the resolution of the model
patch.  After ``cv.resize`` all 33 patches are near-identical, the scale
response is flat, and **the filter silently does nothing while appearing to
run**.  It raises no error and the tracker still reports plausible boxes.

Two defences are built in:

* ``scale_model_max_area`` defaults to 4096, so a target keeps native
  resolution up to 64x64.
* :attr:`ScaleFilter.centre_fraction` reports the fraction of frames whose scale
  response peaked at the centre index (no scale change).  **If that exceeds ~0.9
  on a clip where the target demonstrably changes size, the filter is dead.**
  The benchmark surfaces it.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .correlation import _wrapped_coords, gaussian_label_1d, get_subwindow, resize_patch

__all__ = ["ScaleFilter"]


class ScaleFilter:
    def __init__(
        self,
        target_size: tuple[float, float],
        n_scales: int = 33,
        scale_step: float = 1.02,
        scale_sigma_factor: float = 0.25,
        lr: float = 0.025,
        lam: float = 0.01,
        scale_model_max_area: float = 4096.0,
        min_scale_factor: float = 0.1,
        max_scale_factor: float = 10.0,
        patch_padding: float = 1.0,
    ) -> None:
        self.n_scales = int(n_scales)
        self.scale_step = float(scale_step)
        self.lr = float(lr)
        self.lam = float(lam)

        # Scale exponents in WRAPPED order, so index 0 means "no scale change"
        # and matches the label's peak -- the same convention as the 2-D path.
        self._exponents = _wrapped_coords(self.n_scales).astype(np.float64)
        self.scale_factors = self.scale_step ** self._exponents

        sigma = np.sqrt(self.n_scales) * float(scale_sigma_factor)
        self._label_hat = np.fft.rfft(gaussian_label_1d(self.n_scales, sigma))

        # Hann window ALONG THE SCALE AXIS.  Omitting it makes the response ring
        # badly at the ends of the pyramid.
        self._window = np.hanning(self.n_scales + 2)[1:-1]

        base_w, base_h = float(target_size[0]), float(target_size[1])
        self.base_target_sz = (base_w, base_h)
        # Scale patches include margin so the target's BOUNDARY is always inside
        # the patch.  Extracting at exactly the target size is what canonical
        # DSST does, but it relies on fHOG capturing internal texture; a uniform
        # disc has none, so a patch the size of the disc is simply all-white and
        # most of the scale pool carries no information to discriminate on.  The
        # observable symptom is a filter that tracks scale correctly for one
        # cycle and then cannot recover once the target passes through its
        # minimum size.
        self.patch_padding = float(patch_padding)
        area = base_w * base_h * self.patch_padding * self.patch_padding
        factor = 1.0
        if area > scale_model_max_area:
            factor = np.sqrt(scale_model_max_area / area)
        self.model_sz = (
            max(4, int(round(base_h * self.patch_padding * factor))),
            max(4, int(round(base_w * self.patch_padding * factor))),
        )

        self.current_scale_factor = 1.0
        self.min_scale_factor = float(min_scale_factor)
        self.max_scale_factor = float(max_scale_factor)

        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._n_frames = 0
        self._n_centre = 0

    @property
    def centre_fraction(self) -> float:
        """Fraction of frames whose scale response peaked at "no change".

        The health check for this filter -- see the module docstring.
        """
        return self._n_centre / self._n_frames if self._n_frames else float("nan")

    def _sample(self, gray: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
        """Feature matrix ``(D, n_scales)`` for the current centre and scale."""
        base_w, base_h = self.base_target_sz
        columns = []
        for factor in self.scale_factors:
            size = self.current_scale_factor * factor * self.patch_padding
            patch_h = max(2, int(round(base_h * size)))
            patch_w = max(2, int(round(base_w * size)))
            patch = get_subwindow(gray, centre, (patch_h, patch_w))
            resized = resize_patch(np.asarray(patch, dtype=np.float64), self.model_sz)
            columns.append(resized.ravel())
        feats = np.stack(columns, axis=1)
        feats = feats / 255.0
        feats = feats - feats.mean(axis=0, keepdims=True)
        # Per-column L2 normalisation is NOT optional with intensity features.
        # The scale filter works by detecting a *shift* of the feature matrix
        # along the scale axis when the target changes size.  Raw intensity
        # columns differ in norm by up to 160x across the pool (a small window
        # is nearly all target, a large one is mostly background), and that
        # amplitude variation swamps the shift -- the response then peaks at
        # "no change" on 100% of frames while looking perfectly healthy.
        # Canonical DSST gets this for free because it uses fHOG, whose block
        # normalisation already makes columns comparable.
        feats = feats / (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-9)
        return feats * self._window[None, :]

    def _train(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f_hat = np.fft.rfft(feats, axis=1)
        a = self._label_hat[None, :] * np.conj(f_hat)
        b = np.sum(f_hat * np.conj(f_hat), axis=0).real
        return a, b

    def init(self, gray: np.ndarray, centre: tuple[float, float]) -> None:
        self.current_scale_factor = 1.0
        self._A, self._B = self._train(self._sample(gray, centre))
        self._n_frames = 0
        self._n_centre = 0

    def detect(self, gray: np.ndarray, centre: tuple[float, float]) -> float:
        """Estimate scale at ``centre`` and update ``current_scale_factor``."""
        feats = self._sample(gray, centre)
        z_hat = np.fft.rfft(feats, axis=1)
        num = np.sum(self._A * z_hat, axis=0)
        response = np.fft.irfft(num / (self._B + self.lam), n=self.n_scales)

        idx = int(np.argmax(response))
        self._n_frames += 1
        if idx == 0:
            self._n_centre += 1

        # Parabolic refinement along the scale axis (fDSST does this; the
        # original does not).  Four lines for a visibly smoother trajectory.
        prev = response[(idx - 1) % self.n_scales]
        nxt = response[(idx + 1) % self.n_scales]
        denom = prev - 2.0 * response[idx] + nxt
        delta = 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (prev - nxt) / denom, -0.5, 0.5))

        exponent = float(self._exponents[idx]) + delta
        self.current_scale_factor = float(
            np.clip(
                self.current_scale_factor * (self.scale_step ** exponent),
                self.min_scale_factor,
                self.max_scale_factor,
            )
        )
        self.last_response = response
        self.last_index = idx
        return self.current_scale_factor

    def update(self, gray: np.ndarray, centre: tuple[float, float]) -> None:
        a, b = self._train(self._sample(gray, centre))
        self._A = self.lr * a + (1.0 - self.lr) * self._A
        self._B = self.lr * b + (1.0 - self.lr) * self._B

    def target_size(self) -> tuple[float, float]:
        """Current box size.

        Always recomputed as ``base_target_sz * current_scale_factor`` from the
        immutable init size -- **never** by resizing the previous frame's box.
        That is what stops rounding and aspect drift compounding over thousands
        of frames, and it is the difference between DSST and a naive "multiply
        the box by 1.02 each frame" hack.
        """
        return (
            self.base_target_sz[0] * self.current_scale_factor,
            self.base_target_sz[1] * self.current_scale_factor,
        )
