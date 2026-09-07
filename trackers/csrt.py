"""CSR-DCF-lite -- Lukezic et al. 2017, from scratch.

**Labelling, stated once and prominently: this is CSR-DCF-*lite*.  The paper's
Markov-random-field regularisation of the spatial reliability map is replaced by
morphological close/open plus largest-connected-component selection.**

That substitution is defensible rather than a shortcut: the MRF's job is exactly
spatial coherence and hole filling on a noisy per-pixel posterior, which
close/open + largest-CC accomplishes for a single compact object; a real graph
cut is impossible here (no scipy, no maxflow, and hand-writing
Boykov-Kolmogorov is a project of its own); and **OpenCV's own legacy
``TrackerCSRT`` -- the implementation everybody benchmarks against -- also does
not implement the full MRF**, using a simplified probabilistic segmentation
instead.  So the simplification matches the de facto reference.

The three components, all present:

1. **Spatial reliability map** -- foreground/background histogram backprojection
   times an Epanechnikov spatial prior, then the morphological regularisation
   above, with a safety valve (below).
2. **Constrained filter learning via ADMM** -- 4 iterations.  With a single
   training sample the data term decouples per frequency bin, so the ``h``
   update is elementwise and no Sherman-Morrison is needed.  **Detection uses
   ``g``, the constrained filter -- never ``h``.**
3. **Channel reliability weights** -- learning reliability from each channel's
   own response peak, detection reliability from the ratio of the two major
   response modes, with a floor so one channel cannot capture all the weight.
   *This is why fHOG is mandatory here*: with one channel the whole component is
   identically vacuous.

Scale comes from :class:`trackers.scale.ScaleFilter` verbatim, exactly as the
paper does.

Content-specific notes
----------------------
* The clips are effectively **monochrome** (BGR channels differ by <= 4 in the
  source video), so the colour model is a **1-D grayscale histogram**.  The
  standard HSV recipe would read the undefined hue of grey pixels as noise.
* On a 255 disc over a zero-variance background, backprojection yields an
  essentially perfect mask, so CSRT looks better here than it would on real
  footage.  Worth saying out loud when quoting these numbers.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .base import BaseTracker, TrackerConfig
from .correlation import apce, fft, gaussian_label_2d, get_subwindow, ifft, next_fast_size, peak_subpixel, psr
from .features import FeatureExtractor, hann2d
from .scale import ScaleFilter
from .types import BBox, bbox_from_centre, clip_bbox

__all__ = ["CSRT"]


def _csrt_config(**kwargs) -> TrackerConfig:
    cfg = TrackerConfig(
        # Larger than DSST's 3.0 for the same structural reason (a window of 2x
        # the init box is exactly filled once the target grows 4x, leaving no
        # context), plus one of its own: fHOG at cell_size=4 quantises the
        # response map, so CSRT resolves fewer cells than DSST for an equal
        # window and needs more of it.
        #
        # Measured on synth_scale, sweeping only this parameter:
        #   padding 2.0 -> ScaleErr 0.512, Prec@20 0.693, fails at frame 65
        #   padding 3.0 -> ScaleErr 0.497, Prec@20 0.738, fails at frame 73
        #   padding 4.0 -> ScaleErr 0.122, Prec@20 1.000, never fails
        padding=kwargs.pop("padding", 4.0),
        lr=kwargs.pop("lr", 0.02),
        lam=kwargs.pop("lam", 0.01),
        output_sigma_factor=kwargs.pop("output_sigma_factor", 1.0 / 16.0),
    )
    cfg.extra = {
        "features": kwargs.pop("features", "fhog+gray"),
        "cell_size": kwargs.pop("cell_size", 4),
        "admm_iters": kwargs.pop("admm_iters", 4),
        "mu0": kwargs.pop("mu0", 5.0),
        "beta": kwargs.pop("beta", 3.0),
        "mu_max": kwargs.pop("mu_max", 20.0),
        "hist_bins": kwargs.pop("hist_bins", 32),
        "hist_lr": kwargs.pop("hist_lr", 0.04),
        "mask_threshold": kwargs.pop("mask_threshold", 0.5),
        "weight_floor": kwargs.pop("weight_floor", 0.05),
        "use_scale": kwargs.pop("use_scale", True),
    }
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class CSRT(BaseTracker):
    name = "CSRT"
    type = "appearance"
    supports_scale = True

    def __init__(self, **kwargs) -> None:
        super().__init__(_csrt_config(**kwargs))
        e = self.config.extra
        self.cell_size = int(e["cell_size"])
        self.features = FeatureExtractor(e["features"], cell_size=self.cell_size)
        self.admm_iters = int(e["admm_iters"])
        self.mu0 = float(e["mu0"])
        self.beta = float(e["beta"])
        self.mu_max = float(e["mu_max"])
        self.hist_bins = int(e["hist_bins"])
        self.hist_lr = float(e["hist_lr"])
        self.mask_threshold = float(e["mask_threshold"])
        self.weight_floor = float(e["weight_floor"])
        self.use_scale = bool(e["use_scale"])
        self._g: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self.scale: ScaleFilter | None = None

    # -- colour model -------------------------------------------------------
    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def _histograms(self, gray: np.ndarray, bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
        h, w = gray.shape[:2]
        x, y, bw, bh = clip_bbox(bbox, w, h)
        x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
        fg = gray[y0:y1, x0:x1]

        # Background ring: ~1.6x the box, excluding the interior.
        mx, my = int(bw * 0.3), int(bh * 0.3)
        bx0, by0 = max(0, x0 - mx), max(0, y0 - my)
        bx1, by1 = min(w, x1 + mx), min(h, y1 + my)
        ring = gray[by0:by1, bx0:bx1].copy()
        mask = np.ones(ring.shape, dtype=bool)
        mask[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0] = False
        bg = ring[mask]

        bins = self.hist_bins
        # int64 before the multiply: uint8 * 32 wraps silently.
        fg_idx = (fg.ravel().astype(np.int64) * bins // 256).clip(0, bins - 1)
        bg_idx = (bg.ravel().astype(np.int64) * bins // 256).clip(0, bins - 1)
        fg_hist = np.bincount(fg_idx, minlength=bins).astype(np.float64)
        bg_hist = np.bincount(bg_idx, minlength=bins).astype(np.float64)
        fg_hist /= max(fg_hist.sum(), 1.0)
        bg_hist /= max(bg_hist.sum(), 1.0)
        return fg_hist, bg_hist

    def _spatial_mask(self, gray_patch: np.ndarray) -> np.ndarray:
        """Foreground posterior -> regularised binary mask, at feature resolution.

        The patch is first cropped by one cell on each side.  fHOG drops its
        outer cell ring, so the feature map covers only pixels ``[k, H-k)`` of
        the window; building the mask over the *full* window and resizing it to
        the map misaligns the two by one cell, and the ADMM multiplies the
        filter by this mask.  The crop keeps them aligned.

        The returned mask is **centred** (object in the middle), matching the
        layout of the constrained filter ``g``.
        """
        k = self.cell_size
        if k > 1 and gray_patch.shape[0] > 2 * k and gray_patch.shape[1] > 2 * k:
            gray_patch = gray_patch[k:-k, k:-k]
        bins = self.hist_bins
        idx = (np.asarray(gray_patch, dtype=np.int64) * bins // 256).clip(0, bins - 1)
        fg = self._fg_hist[idx]
        bg = self._bg_hist[idx]
        posterior = fg / (fg + bg + 1e-9)

        # Epanechnikov spatial prior: object pixels are near the centre.
        h, w = gray_patch.shape
        yy, xx = np.mgrid[0:h, 0:w]
        ry = (yy - (h - 1) / 2.0) / (h / 2.0)
        rx = (xx - (w - 1) / 2.0) / (w / 2.0)
        prior = np.clip(1.0 - (rx * rx + ry * ry), 0.0, 1.0)
        posterior = posterior * prior

        peak = posterior.max()
        binary = (posterior >= self.mask_threshold * peak).astype(np.uint8) if peak > 1e-9 else np.zeros((h, w), np.uint8)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

        n, labels, stats, _ = cv.connectedComponentsWithStats(binary, 8)
        if n > 1:
            centre_label = labels[h // 2, w // 2]
            if centre_label == 0:
                centre_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
            binary = (labels == centre_label).astype(np.uint8)

        mask = cv.resize(binary.astype(np.float64), (self._map_shape[1], self._map_shape[0]),
                         interpolation=cv.INTER_AREA)

        # Safety valve.  A degenerate mask means an all-zero filter and total
        # failure, and it WILL degenerate -- e.g. during full occlusion, when the
        # histogram model has nothing to separate.
        frac = float(mask.mean())
        if frac < 0.05 or frac > 0.90:
            mh, mw = self._map_shape
            yy, xx = np.mgrid[0:mh, 0:mw]
            ry = (yy - (mh - 1) / 2.0) / max(mh / 2.0, 1e-9)
            rx = (xx - (mw - 1) / 2.0) / max(mw / 2.0, 1e-9)
            mask = ((rx * rx + ry * ry) <= 1.0).astype(np.float64)
            self._mask_fallback = True
        else:
            self._mask_fallback = False
        return mask

    # -- ADMM ---------------------------------------------------------------
    def _admm(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Constrained filter learning. Returns ``g`` (spatial, ``(C, h, w)``)."""
        shape = self._map_shape
        x_hat = fft(x)
        y_hat = self._label_hat
        lam = self.config.lam
        d = float(shape[0] * shape[1])

        sxy = y_hat[None, ...] * np.conj(x_hat)
        sxx = (x_hat * np.conj(x_hat)).real

        g = np.zeros_like(x)
        l = np.zeros_like(x)
        mu = self.mu0

        # The mask is CENTRED, and so is the constrained filter g: the ADMM's
        # spatial constraint g = m*h confines the filter to the object, and the
        # object sits at the middle of the patch.  (Verified empirically -- an
        # ifftshift here degrades the shift test by an order of magnitude.)
        m = mask[None, ...]

        for _ in range(self.admm_iters):
            g_hat = fft(g)
            l_hat = fft(l)
            h_hat = (sxy + mu * g_hat - l_hat) / (sxx + mu)
            h = ifft(h_hat, shape)
            g = m * ((mu * h + l) / (lam / d + mu))
            l = l + mu * (h - g)
            mu = min(self.mu_max, self.beta * mu)
        return g

    @staticmethod
    def _second_mode(response: np.ndarray, radius: int) -> float:
        """Max outside a window around the global peak -- the second major mode."""
        h, w = response.shape
        iy, ix = np.unravel_index(int(np.argmax(response)), response.shape)
        mask = np.ones((h, w), dtype=bool)
        ys = np.arange(iy - radius, iy + radius + 1) % h
        xs = np.arange(ix - radius, ix + radius + 1) % w
        mask[np.ix_(ys, xs)] = False
        return float(response[mask].max()) if mask.any() else 0.0

    def _channel_weights(self, g: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Learning reliability x detection reliability, normalised, floored."""
        shape = self._map_shape
        # Response of each channel's own filter on its own training patch.
        # No conj here: see the note in update() -- g is already the conjugated
        # filter, matching correlation.py's convention.
        resp = ifft(fft(g) * fft(x), shape)
        learn = resp.reshape(resp.shape[0], -1).max(axis=1)
        learn = np.clip(learn, 0.0, None)

        radius = max(1, int(0.15 * min(shape)))
        detect = np.empty(resp.shape[0], dtype=np.float64)
        for c in range(resp.shape[0]):
            r = resp[c]
            peak = float(r.max())
            second = self._second_mode(r, radius)
            ratio = (second / peak) if peak > 1e-9 else 1.0
            detect[c] = 1.0 - min(ratio, 0.5) / 0.5

        w = learn * detect
        if w.sum() <= 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        # Floor: without it a single channel can capture all the weight and the
        # response becomes that channel's noise.
        w = np.maximum(w, self.weight_floor * w.mean())
        return w / w.sum()

    # -- interface ----------------------------------------------------------
    def _extract(self, frame: np.ndarray, centre: tuple[float, float],
                 scale: float) -> tuple[np.ndarray, np.ndarray]:
        size = (max(4, int(round(self._window[0] * scale))),
                max(4, int(round(self._window[1] * scale))))
        patch = get_subwindow(frame, centre, size)
        if size != self._window:
            patch = cv.resize(patch, (self._window[1], self._window[0]),
                              interpolation=cv.INTER_AREA if size[0] > self._window[0] else cv.INTER_LINEAR)
        return self.features(patch), self._gray(patch)

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        self._reset_state(bbox)
        if frame.ndim == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        pad = 1.0 + self.config.padding
        k = self.cell_size
        self._window = (
            int(round(bbox[3] * pad / k + 2)) * k,
            int(round(bbox[2] * pad / k + 2)) * k,
        )

        self._fg_hist, self._bg_hist = self._histograms(self._gray(frame), bbox)

        x, gray_patch = self._extract(frame, self._centre, 1.0)
        self._map_shape = x.shape[1:]
        sigma = np.sqrt(bbox[2] * bbox[3]) * self.config.output_sigma_factor / k
        self._label_hat = fft(gaussian_label_2d(self._map_shape, float(sigma)))

        mask = self._spatial_mask(gray_patch)
        self._mask = mask
        self._g = self._admm(x, mask)
        self._weights = self._channel_weights(self._g, x)

        self.scale = None
        if self.use_scale:
            self.scale = ScaleFilter(target_size=(float(bbox[2]), float(bbox[3])))
            self.scale.init(self._gray(frame).astype(np.float64), self._centre)
        self.diagnostics = {"window": self._window, "map": self._map_shape,
                            "mask_area_frac": float(mask.mean())}

    def update(self, frame: np.ndarray) -> tuple[bool, BBox]:
        if frame.ndim == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        sf = self.scale.current_scale_factor if self.scale else 1.0

        z, gray_patch = self._extract(frame, self._centre, sf)
        # Detection uses g, the CONSTRAINED filter -- never h.
        #
        # No conj on g.  The ADMM builds h from ``y_hat * conj(x_hat)``, so g is
        # already the *conjugated* filter (Bolme's H*), exactly as
        # correlation.py stores filters.  Applying conj again here double-
        # conjugates.  On a centrally symmetric target that is a near-identity
        # reflection, so it does not mirror the track outright -- it surfaced
        # only as a constant one-cell position offset, which is precisely the
        # kind of bug the fixed convention exists to prevent.
        per_channel = ifft(fft(self._g) * fft(z), self._map_shape)
        response = np.tensordot(self._weights, per_channel, axes=(0, 0))

        dy, dx, peak = peak_subpixel(response)
        dy *= self.cell_size * sf
        dx *= self.cell_size * sf
        cx = float(np.clip(self._centre[0] + dx, 0, w - 1))
        cy = float(np.clip(self._centre[1] + dy, 0, h - 1))
        self._centre = (cx, cy)

        raw_psr = psr(response, min(11, max(3, min(self._map_shape) // 3)))
        conf = self._record_confidence(raw_psr)

        if self.scale is not None:
            self.scale.detect(self._gray(frame).astype(np.float64), self._centre)
            self._size = self.scale.target_size()

        if self._should_update(conf):
            sf = self.scale.current_scale_factor if self.scale else 1.0
            x_new, gray_new = self._extract(frame, self._centre, sf)
            fg, bg = self._histograms(self._gray(frame), bbox_from_centre(cx, cy, *self._size))
            self._fg_hist = (1 - self.hist_lr) * self._fg_hist + self.hist_lr * fg
            self._bg_hist = (1 - self.hist_lr) * self._bg_hist + self.hist_lr * bg

            mask = self._spatial_mask(gray_new)
            self._mask = mask
            g_new = self._admm(x_new, mask)
            lr = self.config.lr
            self._g = (1.0 - lr) * self._g + lr * g_new
            self._weights = self._channel_weights(self._g, x_new)
            if self.scale is not None:
                self.scale.update(self._gray(frame).astype(np.float64), self._centre)

        self.diagnostics = {
            "psr": raw_psr, "apce": apce(response), "peak": peak,
            "mask_area_frac": float(self._mask.mean()),
            "mask_fallback": self._mask_fallback,
            "channel_weights": self._weights,
            "scale_factor": sf,
            "updated": conf >= self.config.conf_ratio,
        }
        if self.config.debug:
            self.diagnostics["response"] = response
            self.diagnostics["mask"] = self._mask

        return (not self._lost), clip_bbox(bbox_from_centre(cx, cy, *self._size), w, h)
