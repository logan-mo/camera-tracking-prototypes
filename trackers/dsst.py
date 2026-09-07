"""DSST -- Danelljan et al. 2014, from scratch.

The direct fix for the scale-variation failure: a 2-D translation filter **plus
a completely separate 1-D scale filter** (:mod:`trackers.scale`).

DSST's translation filter is *not* KCF.  It is a linear multi-channel MOSSE --
primal, per-channel numerator, one scalar denominator shared across channels --
so it reuses ``dcf_train``/``dcf_detect`` from :mod:`trackers.correlation`, the
same code MOSSE calls with ``C=1``.  KCF (kernelised) and CSRT (ADMM) do not.

Order of operations per frame, which is easy to get subtly wrong
----------------------------------------------------------------
1. Extract the translation patch at the **previous** centre, at size
   ``base_window * current_scale_factor``, then resize back to ``base_window``.
   The translation filter always works at a *fixed model resolution*; scale is
   handled entirely by changing the extraction size.
2. Translation response -> subpixel peak -> shift in model coords ->
   **multiply by ``current_scale_factor``** -> new centre.
3. At the **new** centre, run the scale filter -> new ``current_scale_factor``.
4. Update the translation filter at the new centre and new scale.
5. Update the scale filter.

Step 2's multiplication is the usual culprit when DSST tracks but lags.  To
isolate a bug: run with the translation step replaced by ground-truth centres
(free from the sidecar).  If scale still fails with a perfect centre, the bug is
in the scale filter; if it works, the bug is in the coupling.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .base import BaseTracker, TrackerConfig
from .correlation import (
    apce,
    dcf_detect,
    dcf_train,
    fft,
    gaussian_label_2d,
    get_subwindow,
    next_fast_size,
    peak_subpixel,
    psr,
    resize_patch,
    subwindow_centre,
)
from .features import hann2d
from .scale import ScaleFilter
from .types import BBox, bbox_from_centre, clip_bbox

__all__ = ["DSST"]


def _dsst_config(**kwargs) -> TrackerConfig:
    cfg = TrackerConfig(
        # Canonical DSST uses padding=1.0.  That is too tight here, and the
        # reason is structural rather than a fudge: the scale clip grows the
        # target 4x linearly (12->48 px radius), so a window of 2x the *init*
        # box is exactly filled by the target at maximum scale, leaving no
        # context for the correlation peak.  Any centre error is then
        # unrecoverable, the scale filter falls behind, the window shrinks with
        # it, and the two feed each other.
        #
        # Measured on synth_scale, sweeping only this parameter:
        #   padding 1.0 -> ScaleErr 0.478, Prec@20 0.710, fails at frame 375
        #   padding 2.0 -> ScaleErr 0.306, Prec@20 0.873, fails at frame 750
        #   padding 3.0 -> ScaleErr 0.084, Prec@20 1.000, never fails
        # 0.084 matches what the scale filter achieves when fed ground-truth
        # centres (0.087), i.e. at this padding the translation step stops being
        # the bottleneck.  Cost is ~5% of throughput.
        padding=kwargs.pop("padding", 3.0),
        lr=kwargs.pop("lr", 0.025),
        lam=kwargs.pop("lam", 0.01),
        output_sigma_factor=kwargs.pop("output_sigma_factor", 1.0 / 16.0),
    )
    cfg.extra = {
        "n_scales": kwargs.pop("n_scales", 33),
        "scale_step": kwargs.pop("scale_step", 1.02),
        "scale_lr": kwargs.pop("scale_lr", 0.025),
        "scale_lam": kwargs.pop("scale_lam", 0.01),
        "scale_model_max_area": kwargs.pop("scale_model_max_area", 4096.0),
        "scale_sigma_factor": kwargs.pop("scale_sigma_factor", 0.25),
    }
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class DSST(BaseTracker):
    name = "DSST"
    type = "appearance"
    supports_scale = True

    def __init__(self, **kwargs) -> None:
        super().__init__(_dsst_config(**kwargs))
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self.scale: ScaleFilter | None = None

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return (frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)).astype(
            np.float64
        )

    def _extract_size(self) -> tuple[int, int]:
        sf = self.scale.current_scale_factor if self.scale else 1.0
        return (max(2, int(round(self._window[0] * sf))), max(2, int(round(self._window[1] * sf))))

    def _translation_patch(self, gray: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
        """Patch at the current scale, resized back to the fixed model window."""
        size = self._extract_size()
        patch = get_subwindow(gray, centre, size)
        if size != self._window:
            patch = resize_patch(patch, self._window)
        x = patch / 255.0
        x = x - x.mean()
        # L2-normalise, as MOSSE's preprocessing does.  Without it the patch
        # energy changes as the target grows or shrinks, so the running A/B
        # averages are accumulated at a drifting gain and the filter's effective
        # sharpness varies with scale -- which is exactly the regime DSST spends
        # its whole life in.
        x = x / (np.linalg.norm(x) + 1e-9)
        return (x * self._hann)[None, ...]

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._reset_state(bbox)
        gray = self._gray(frame)

        pad = 1.0 + self.config.padding
        self._window = (
            next_fast_size(int(round(bbox[3] * pad))),
            next_fast_size(int(round(bbox[2] * pad))),
        )
        self._hann = hann2d(self._window)
        sigma = np.sqrt(bbox[2] * bbox[3]) * self.config.output_sigma_factor
        self._label_hat = fft(gaussian_label_2d(self._window, float(sigma)))

        e = self.config.extra
        self.scale = ScaleFilter(
            target_size=(float(bbox[2]), float(bbox[3])),
            n_scales=int(e["n_scales"]),
            scale_step=float(e["scale_step"]),
            scale_sigma_factor=float(e["scale_sigma_factor"]),
            lr=float(e["scale_lr"]),
            lam=float(e["scale_lam"]),
            scale_model_max_area=float(e["scale_model_max_area"]),
        )

        x_hat = fft(self._translation_patch(gray, self._centre))
        self._A, self._B = dcf_train(x_hat, self._label_hat)
        self.scale.init(gray, self._centre)
        self.diagnostics = {"window": self._window, "scale_model": self.scale.model_sz}

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        gray = self._gray(frame)
        h, w = gray.shape[:2]
        assert self.scale is not None

        # 1-2. Translation.  The shift is measured relative to the patch, so it
        # is added to the patch's effective centre, not to the requested one.
        size = self._extract_size()
        eff = subwindow_centre(self._centre, size)
        z_hat = fft(self._translation_patch(gray, self._centre))
        response = dcf_detect(self._A, self._B, z_hat, self.config.lam, self._window)
        dy, dx, peak = peak_subpixel(response)
        sf = self.scale.current_scale_factor
        cx = float(np.clip(eff[0] + dx * sf, 0, w - 1))
        cy = float(np.clip(eff[1] + dy * sf, 0, h - 1))
        self._centre = (cx, cy)

        raw_psr = psr(response, 11)
        conf = self._record_confidence(raw_psr)

        # 3. Scale, at the new centre.
        self.scale.detect(gray, self._centre)
        self._size = self.scale.target_size()

        # 4-5. Update both filters at the new centre and scale.
        if self._should_update(conf):
            x_hat = fft(self._translation_patch(gray, self._centre))
            a, b = dcf_train(x_hat, self._label_hat)
            lr = self.config.lr
            self._A = lr * a + (1.0 - lr) * self._A
            self._B = lr * b + (1.0 - lr) * self._B
            self.scale.update(gray, self._centre)

        self.diagnostics = {
            "psr": raw_psr, "apce": apce(response), "peak": peak,
            "scale_factor": self.scale.current_scale_factor,
            "scale_index": self.scale.last_index,
            "scale_centre_fraction": self.scale.centre_fraction,
            "updated": conf >= self.config.conf_ratio,
        }
        if self.config.debug:
            self.diagnostics["response"] = response
            self.diagnostics["scale_response"] = self.scale.last_response

        return (not self._lost), clip_bbox(bbox_from_centre(cx, cy, *self._size), w, h)
