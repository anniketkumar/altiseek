"""The detector inference API (CLAUDE.md §9 contract).

Input: a batch of ``R x R`` uint8 RGB views. Output per view: detections as (class id, raw score,
box in view pixel-corner coordinates), from a checkpoint identified by ``detector_id``, the
leading 16 hex digits of the checkpoint file's SHA-256 (§5). Deterministic apart from GPU
nondeterminism. Ultralytics' usage analytics are switched off.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from altiseek.registry.run import sha256_file


@dataclass(frozen=True)
class Detection:
    """One detection in a view; the box is in view pixel-corner coordinates."""

    class_id: int
    score: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def detector_id_of(checkpoint: str | Path) -> str:
    """Content ID of a checkpoint file."""
    return sha256_file(checkpoint)[:16]


class Detector:
    """A trained Ultralytics checkpoint behind the AltiSeek inference contract."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        score_min: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> None:
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        self.checkpoint = Path(checkpoint)
        self.detector_id = detector_id_of(self.checkpoint)
        self.device = device
        self.score_min = score_min
        self.iou = iou
        self.max_det = max_det
        self._model = YOLO(str(self.checkpoint))

    def detect(self, views: NDArray[np.uint8]) -> list[list[Detection]]:
        """Detections for each view of an ``N x R x R x 3`` RGB batch."""
        if views.ndim != 4 or views.shape[1] != views.shape[2] or views.shape[3] != 3:
            raise ValueError(f"expected N x R x R x 3 views, got {views.shape}")
        if len(views) == 0:
            return []
        bgr = [np.ascontiguousarray(v[..., ::-1]) for v in views]  # Ultralytics: arrays are BGR
        results = self._model.predict(
            bgr,
            imgsz=views.shape[1],
            conf=self.score_min,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        out = []
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            out.append(
                [
                    Detection(int(c), float(s), *(float(x) for x in box))
                    for c, s, box in zip(classes, scores, xyxy, strict=True)
                ]
            )
        return out
