from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.vision.detector import YoloCapDetector


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Row:
    def tolist(self) -> list[float]:
        return [10.0, 20.0, 30.0, 40.0]


class _Boxes:
    cls = [_Scalar(0)]
    conf = [_Scalar(0.91)]
    xyxy = [_Row()]

    def __len__(self) -> int:
        return 1


class _Result:
    boxes = _Boxes()


class _Model:
    names = {0: "tube_cap"}

    def predict(self, **kwargs: object) -> list[_Result]:
        return [_Result()]


class YoloCapDetectorTests(unittest.TestCase):
    def test_model_import_and_load_are_lazy(self) -> None:
        detector = YoloCapDetector(
            "model_does_not_need_to_exist_yet.pt",
            {0: "tube_cap"},
            0.45,
            0.45,
            1280,
            "cpu",
        )

        fake_module = types.ModuleType("ultralytics")
        fake_module.YOLO = lambda path: _Model()
        with patch.dict(sys.modules, {"ultralytics": fake_module}):
            detections = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "tube_cap")
        self.assertEqual(detections[0].box.center.u, 20.0)
        self.assertEqual(detections[0].box.center.v, 30.0)

    def test_wrong_class_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(VisionError, "YOLO classes"):
            YoloCapDetector(
                "model.pt",
                {0: "cap"},
                0.45,
                0.45,
                1280,
                "cpu",
            )


if __name__ == "__main__":
    unittest.main()
