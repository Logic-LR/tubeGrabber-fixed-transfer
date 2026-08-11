"""Small Ultralytics YOLO adapter.

The import and model load are delayed until the first call to ``detect`` so
the rest of the project can run without Ultralytics or a model file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Box, Detection


EXPECTED_CLASSES = {0: "tube_cap"}


class YoloCapDetector:
    """Run the one-class tube-cap detector used by the final runtime."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: Mapping[int, str],
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
    ) -> None:
        normalized_names = {int(key): str(value) for key, value in class_names.items()}
        if normalized_names != EXPECTED_CLASSES:
            raise VisionError(
                f"YOLO classes must be {EXPECTED_CLASSES}"
            )
        if not 0.0 < confidence <= 1.0:
            raise VisionError("YOLO confidence must be between 0 and 1")
        if not 0.0 < iou <= 1.0:
            raise VisionError("YOLO IoU must be between 0 and 1")
        if image_size <= 0:
            raise VisionError("YOLO image_size must be positive")

        self._model_path = str(model_path)
        self._class_names = normalized_names
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._image_size = int(image_size)
        self._device = str(device)
        self._model: object | None = None

    def detect(self, color_image: object) -> list[Detection]:
        image = np.asarray(color_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("YOLO input must be a BGR image with three channels")

        model = self._get_model()
        try:
            results = model.predict(
                source=image,
                conf=self._confidence,
                iou=self._iou,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            raise VisionError(f"YOLO inference failed: {exc}") from exc

        if not results or results[0].boxes is None:
            return []

        detections: list[Detection] = []
        boxes = results[0].boxes
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            if class_id not in self._class_names:
                raise VisionError(f"YOLO returned unsupported class id {class_id}")
            x1, y1, x2, y2 = (
                float(value) for value in boxes.xyxy[index].tolist()
            )
            detections.append(
                Detection(
                    label=self._class_names[class_id],
                    confidence=float(boxes.conf[index].item()),
                    box=Box(x1, y1, x2, y2),
                )
            )
        return detections

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionError(
                "Ultralytics is not installed; install the optional vision dependency"
            ) from exc

        try:
            model = YOLO(self._model_path)
        except Exception as exc:
            raise VisionError(f"cannot load YOLO model {self._model_path}: {exc}") from exc

        model_names = {
            int(key): str(value) for key, value in dict(model.names).items()
        }
        if model_names != self._class_names:
            raise VisionError(
                f"model classes are {model_names}, expected {self._class_names}"
            )

        self._model = model
        return model
