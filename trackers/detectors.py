"""Per-frame detectors.

``SORT`` consumes ``list[Detection]`` and nothing else, and it imports
``Detection`` from :mod:`trackers.types` rather than from here.  That is the
whole contract: swapping ``BlobDetector`` for a YOLO backend requires no change
to ``sort.py``, ``runner.py`` or ``benchmark.py`` -- only a different
``--detector`` spec string.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2 as cv
import numpy as np

from .types import BBox, Detection

__all__ = ["BlobDetector", "Detector", "YoloDetector", "detector_from_spec"]

# A filled disc has extent (area / bbox area) = pi/4.
_DISC_EXTENT = np.pi / 4.0


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class BlobDetector:
    """Threshold + contour detector for bright blobs on a dark background.

    Defaults are tuned against ``tracking_test_video.mp4`` and the generated
    synthetic clips, where the background is a uniform grey 15 and the target is
    a white disc:

    ``min_area=200``
        Admits the scale clip's smallest circle (r=12 -> area 452) with better
        than 2x headroom.

    ``min_extent=0.5``
        Rejects the static arrow/pointer overlay that appears in two runs of the
        source clip (frames 1393-1449 and 2822-2834).  That blob has a 69x60
        bbox but only ~57 px^2 of contour area, so its extent is ~0.014 against
        the disc's ~0.72.  With this predicate the source clip yields exactly
        one candidate on 2834 of its 2835 frames.
    """

    name = "blob"

    def __init__(
        self,
        thresh: int = 200,
        min_area: float = 200.0,
        max_area: float | None = None,
        min_extent: float = 0.5,
        invert: bool = False,
        blur: int = 0,
    ) -> None:
        self.thresh = int(thresh)
        self.min_area = float(min_area)
        self.max_area = None if max_area is None else float(max_area)
        self.min_extent = float(min_extent)
        self.invert = bool(invert)
        self.blur = int(blur)

    def params(self) -> dict[str, object]:
        """Parameters that affect output -- used to key the ground-truth cache."""
        return {
            "name": self.name,
            "thresh": self.thresh,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_extent": self.min_extent,
            "invert": self.invert,
            "blur": self.blur,
        }

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.blur > 0:
            k = self.blur * 2 + 1
            gray = cv.GaussianBlur(gray, (k, k), 0)

        mode = cv.THRESH_BINARY_INV if self.invert else cv.THRESH_BINARY
        _, binary = cv.threshold(gray, self.thresh, 255, mode)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        out: list[Detection] = []
        for contour in contours:
            area = float(cv.contourArea(contour))
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            x, y, w, h = cv.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            extent = area / float(w * h)
            if extent < self.min_extent:
                continue
            bbox: BBox = (float(x), float(y), float(w), float(h))
            out.append(Detection(bbox=bbox, score=self._score(extent), class_id=0))
        return out

    @staticmethod
    def _score(extent: float) -> float:
        """Shape-plausibility proxy in [0, 1]; 1.0 for a perfect filled disc.

        A blob has no learned confidence.  This exists so SORT has something to
        gate on via ``min_score``; it is deliberately NOT a quality ranking and
        must not be presented as a detection probability.
        """
        return float(np.clip(extent / _DISC_EXTENT, 0.0, 1.0))


class YoloDetector:
    """ONNX YOLO backend via ``cv.dnn``.

    Deliberately unimplemented -- it is a documented seam, not dead code.
    ``cv.dnn`` *is* present in opencv-python 5.0, so this needs only an .onnx
    file and a decode step; no ultralytics, no torch.  Sketch::

        blob = cv.dnn.blobFromImage(frame, 1/255., (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()          # (1, 84, 8400) for v8
        # decode -> boxes/scores, cv.dnn.NMSBoxes, rescale to frame coords

    Because the CLI takes a spec string parsed by :func:`detector_from_spec`,
    adding this changes no other module.
    """

    name = "yolo"

    def __init__(self, model_path: str, conf: float = 0.25, nms: float = 0.45) -> None:
        self.model_path = model_path
        self.conf = conf
        self.nms = nms
        raise NotImplementedError(
            "YoloDetector is a documented seam, not yet implemented. "
            "Supply an .onnx model and decode cv.dnn output into Detection objects; "
            "no other module needs to change."
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError


def detector_from_spec(spec: str) -> Detector:
    """Build a detector from ``"blob"``, ``"blob:thresh=180,min_area=100"`` or
    ``"yolo:models/yolov8n.onnx"``."""
    spec = (spec or "blob").strip()
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()

    if kind == "blob":
        kwargs: dict[str, object] = {}
        for item in filter(None, (p.strip() for p in rest.split(","))):
            key, _, value = item.partition("=")
            key = key.strip()
            value = value.strip()
            if key in {"thresh", "blur"}:
                kwargs[key] = int(value)
            elif key in {"min_area", "max_area", "min_extent"}:
                kwargs[key] = float(value)
            elif key == "invert":
                kwargs[key] = value.lower() in {"1", "true", "yes"}
            else:
                raise ValueError(f"unknown blob detector option: {key!r}")
        return BlobDetector(**kwargs)  # type: ignore[arg-type]

    if kind == "yolo":
        if not rest:
            raise ValueError("yolo detector spec needs a model path: yolo:<path.onnx>")
        return YoloDetector(rest.strip())

    raise ValueError(f"unknown detector kind: {kind!r} (expected 'blob' or 'yolo')")
