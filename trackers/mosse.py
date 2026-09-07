"""MOSSE -- Bolme et al. 2010, from scratch.

Deliberately the worst case in this bench: single channel, no kernel, no scale,
**fixed box**.  It is here to establish the floor that DSST and CSRT have to
beat, and to show what "fastest OpenCV tracker" actually buys you.

    A_t = eta * F(y) * conj(F(x_t)) + (1 - eta) * A_{t-1}
    B_t = eta * F(x_t) * conj(F(x_t)) + (1 - eta) * B_{t-1}
    response = irfft2( A / (B + eps) * F(z) )

Two details that matter more than they look:

* **Random affine perturbations at init.**  Without them the filter is a matched
  filter to a single image and generalises poorly for the first ~20 frames.
* **``eps`` is relative** to ``mean(|F(x)|^2)``, not an absolute constant.  An
  absolute regulariser is meaningless once preprocessing changes the scale of
  the spectrum.
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
    psr,
)
from .features import hann2d, preprocess_mosse
from .types import BBox, bbox_from_centre, clip_bbox

__all__ = ["MOSSE", "MosseConfig"]


class MosseConfig(TrackerConfig):
    pass


def _mosse_config(**kwargs) -> TrackerConfig:
    cfg = TrackerConfig(
        padding=kwargs.pop("padding", 1.0),
        lr=kwargs.pop("lr", 0.125),
        lam=kwargs.pop("lam", 1e-5),
    )
    cfg.extra = {
        "output_sigma": kwargs.pop("output_sigma", 2.0),
        "n_perturb": kwargs.pop("n_perturb", 8),
        "use_log": kwargs.pop("use_log", True),
        "psr_exclude": kwargs.pop("psr_exclude", 11),
    }
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class MOSSE(BaseTracker):
    name = "MOSSE"
    type = "appearance"
    supports_scale = False

    def __init__(self, **kwargs) -> None:
        super().__init__(_mosse_config(**kwargs))
        e = self.config.extra
        self.output_sigma = float(e["output_sigma"])
        self.n_perturb = int(e["n_perturb"])
        self.use_log = bool(e["use_log"])
        self.psr_exclude = int(e["psr_exclude"])
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def _patch(self, gray: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
        raw = get_subwindow(gray, centre, self._window)
        return preprocess_mosse(raw, self._hann, use_log=self.use_log)

    def _perturb(self, raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Small affine warp about the patch centre, so the label stays fixed."""
        h, w = raw.shape[:2]
        angle = rng.uniform(-3.0, 3.0)
        scale = 1.0 + rng.uniform(-0.02, 0.02)
        M = cv.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += rng.uniform(-1.0, 1.0)
        M[1, 2] += rng.uniform(-1.0, 1.0)
        return cv.warpAffine(raw, M, (w, h), flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REFLECT)

    # -- interface ----------------------------------------------------------
    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._reset_state(bbox)
        gray = self._gray(frame).astype(np.float64)

        pad = 1.0 + self.config.padding
        self._window = (
            next_fast_size(int(round(bbox[3] * pad))),
            next_fast_size(int(round(bbox[2] * pad))),
        )
        self._hann = hann2d(self._window)
        self._label_hat = fft(gaussian_label_2d(self._window, self.output_sigma))

        raw = get_subwindow(gray, self._centre, self._window)
        rng = np.random.default_rng(0)  # seeded: the bench asserts determinism

        A = None
        B = None
        for i in range(max(1, self.n_perturb)):
            sample = raw if i == 0 else self._perturb(raw, rng)
            x_hat = fft(preprocess_mosse(sample, self._hann, use_log=self.use_log))
            a, b = dcf_train(x_hat, self._label_hat)
            A = a if A is None else A + a
            B = b if B is None else B + b
        self._A, self._B = A, B
        self._eps = float(np.mean(np.abs(self._B))) * self.config.lam + 1e-12
        self.diagnostics = {"window": self._window}

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        gray = self._gray(frame).astype(np.float64)
        h, w = gray.shape[:2]

        z_hat = fft(self._patch(gray, self._centre))
        response = dcf_detect(self._A, self._B, z_hat, self._eps, self._window)

        from .correlation import peak_subpixel

        dy, dx, peak = peak_subpixel(response)
        raw_psr = psr(response, self.psr_exclude)
        conf = self._record_confidence(raw_psr)

        cx = float(np.clip(self._centre[0] + dx, 0, w - 1))
        cy = float(np.clip(self._centre[1] + dy, 0, h - 1))
        self._centre = (cx, cy)

        if self._should_update(conf):
            x_hat = fft(self._patch(gray, self._centre))
            a, b = dcf_train(x_hat, self._label_hat)
            lr = self.config.lr
            self._A = lr * a + (1.0 - lr) * self._A
            self._B = lr * b + (1.0 - lr) * self._B

        self.diagnostics = {
            "psr": raw_psr, "apce": apce(response), "peak": peak,
            "peak_xy": (dx, dy), "updated": conf >= self.config.conf_ratio,
        }
        if self.config.debug:
            self.diagnostics["response"] = response

        bbox = clip_bbox(bbox_from_centre(cx, cy, *self._size), w, h)
        return (not self._lost), bbox
