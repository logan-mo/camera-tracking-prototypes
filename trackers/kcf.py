"""KCF -- Henriques et al. 2015, from scratch.

The idea in one paragraph: KCF never explicitly extracts negative samples.  All
cyclic shifts of the padded patch are *implicit* training samples, and because
the resulting data matrix is circulant it is diagonalised by the DFT -- which
collapses the whole kernel ridge regression to one elementwise division:

    k_xx  = gaussian_correlation(x, x)
    alpha = F(y) / (F(k_xx) + lambda)
    r     = irfft2( F(k_xz) * alpha )

Gaussian kernel correlation, with ``N`` the per-channel element count and ``C``
the channel count:

    k = exp( -1/sigma^2 * max(0, ||x||^2 + ||z||^2 - 2*irfft2(sum_c Z_c conj(X_c))) / (N*C) )

The ``max(0, .)`` clamp is not cosmetic: floating-point error makes the argument
slightly negative at the peak, and without the clamp ``k > 1`` there and the
dual is subtly wrong.

The failure mode this tracker exists to demonstrate
---------------------------------------------------
KCF has **no scale estimation** -- the box size is frozen at its init value.  On
a centrally symmetric target like a disc, that failure is *invisible in centre
error*: the correlation peak stays well-centred as the target grows, and only
the box is wrong.  It shows up **only in IoU and ScaleErr**.  That is why the
results table reports those separately, and it is the single most important
measurement decision in this bench.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .base import BaseTracker, TrackerConfig
from .correlation import (
    apce,
    fft,
    gaussian_label_2d,
    get_subwindow,
    ifft,
    next_fast_size,
    peak_subpixel,
    psr,
)
from .features import FeatureExtractor
from .types import BBox, bbox_from_centre, clip_bbox

__all__ = ["KCF", "gaussian_correlation"]


def gaussian_correlation(x_hat: np.ndarray, z_hat: np.ndarray, x: np.ndarray,
                         z: np.ndarray, sigma: float,
                         shape: tuple[int, int]) -> np.ndarray:
    """Gaussian kernel correlation between feature maps ``x`` and ``z``."""
    channels = x.shape[0]
    n = shape[0] * shape[1]
    cross = ifft(np.sum(z_hat * np.conj(x_hat), axis=0), shape)
    energy = float(np.vdot(x.ravel(), x.ravel()).real) + float(np.vdot(z.ravel(), z.ravel()).real)
    d = np.maximum(0.0, (energy - 2.0 * cross) / (n * channels))
    return np.exp(-d / (sigma * sigma))


def _kcf_config(**kwargs) -> TrackerConfig:
    features = kwargs.pop("features", "gray")
    cell = 4 if features.startswith("fhog") else 1
    cfg = TrackerConfig(
        padding=kwargs.pop("padding", 1.5),
        lr=kwargs.pop("lr", 0.075 if cell == 1 else 0.02),
        lam=kwargs.pop("lam", 1e-4),
        output_sigma_factor=kwargs.pop("output_sigma_factor", 0.1),
    )
    cfg.extra = {
        "features": features,
        "cell_size": kwargs.pop("cell_size", cell),
        "kernel_sigma": kwargs.pop("kernel_sigma", 0.2 if cell == 1 else 0.5),
    }
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class KCF(BaseTracker):
    name = "KCF"
    type = "appearance"
    supports_scale = False

    def __init__(self, **kwargs) -> None:
        super().__init__(_kcf_config(**kwargs))
        e = self.config.extra
        self.cell_size = int(e["cell_size"])
        self.kernel_sigma = float(e["kernel_sigma"])
        self.features = FeatureExtractor(e["features"], cell_size=self.cell_size)
        if e["features"] != "gray":
            self.name = f"KCF-{e['features']}"
        self._alpha_hat: np.ndarray | None = None
        self._model: np.ndarray | None = None

    @staticmethod
    def _bgr(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 3 else cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    def _extract(self, frame: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
        patch = get_subwindow(frame, centre, self._window)
        return self.features(patch)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._reset_state(bbox)
        frame = self._bgr(frame)

        pad = 1.0 + self.config.padding
        # Size the window so that, after fHOG drops its outer cell ring, the
        # feature map is the size we intended -- a guaranteed off-by-one
        # otherwise.
        wh = int(round(bbox[2] * pad))
        ht = int(round(bbox[3] * pad))
        if self.cell_size > 1:
            wh = int(round(wh / self.cell_size + 2)) * self.cell_size
            ht = int(round(ht / self.cell_size + 2)) * self.cell_size
        else:
            wh, ht = next_fast_size(wh), next_fast_size(ht)
        self._window = (ht, wh)

        x = self._extract(frame, self._centre)
        self._map_shape = x.shape[1:]

        sigma = (
            np.sqrt(bbox[2] * bbox[3]) * self.config.output_sigma_factor / self.cell_size
        )
        self._label_hat = fft(gaussian_label_2d(self._map_shape, float(sigma)))

        self._model = x
        x_hat = fft(x)
        k = gaussian_correlation(x_hat, x_hat, x, x, self.kernel_sigma, self._map_shape)
        self._alpha_hat = self._label_hat / (fft(k) + self.config.lam)
        self.diagnostics = {"window": self._window, "map": self._map_shape}

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        frame = self._bgr(frame)
        h, w = frame.shape[:2]

        z = self._extract(frame, self._centre)
        z_hat = fft(z)
        x_hat = fft(self._model)
        k = gaussian_correlation(
            x_hat, z_hat, self._model, z, self.kernel_sigma, self._map_shape
        )
        response = ifft(self._alpha_hat * fft(k), self._map_shape)

        dy, dx, peak = peak_subpixel(response)
        # Feature-map cells are cell_size pixels wide; forgetting this makes the
        # tracker lag at exactly 1/cell_size of the true velocity.
        dy *= self.cell_size
        dx *= self.cell_size

        raw_psr = psr(response, min(11, max(3, min(self._map_shape) // 3)))
        conf = self._record_confidence(raw_psr)

        cx = float(np.clip(self._centre[0] + dx, 0, w - 1))
        cy = float(np.clip(self._centre[1] + dy, 0, h - 1))
        self._centre = (cx, cy)

        if self._should_update(conf):
            x_new = self._extract(frame, self._centre)
            xf = fft(x_new)
            k_new = gaussian_correlation(
                xf, xf, x_new, x_new, self.kernel_sigma, self._map_shape
            )
            # alpha_new comes from the NEW patch's own k_xx, not from the
            # interpolated template -- both variants exist in the wild; the
            # reference MATLAB does this one.
            alpha_new = self._label_hat / (fft(k_new) + self.config.lam)
            lr = self.config.lr
            self._alpha_hat = (1.0 - lr) * self._alpha_hat + lr * alpha_new
            self._model = (1.0 - lr) * self._model + lr * x_new

        self.diagnostics = {
            "psr": raw_psr, "apce": apce(response), "peak": peak,
            "peak_xy": (dx, dy), "updated": conf >= self.config.conf_ratio,
        }
        if self.config.debug:
            self.diagnostics["response"] = response

        return (not self._lost), clip_bbox(bbox_from_centre(cx, cy, *self._size), w, h)
