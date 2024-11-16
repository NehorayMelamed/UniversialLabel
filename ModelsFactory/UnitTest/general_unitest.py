from typing import Dict, List

import numpy as np
import cv2
import os
from abc import ABC, abstractmethod
import unittest

from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel


# Unit Test Class for Segmentation and Detection Models
class TestBaseModel(unittest.TestCase):
    def test_segmentation_model(self):
        class MockSegmentationModel(SegmentationBaseModel):
            def init_model(self):
                pass

            def set_prompt(self, prompt: str):
                self.prompt = prompt

            def set_image(self, image):
                self.image = image

            def get_result(self):
                pass

            def get_masks(self):
                return np.zeros((640, 640), dtype=np.uint8)  # Mock mask output

        # Instantiate and test the segmentation model
        segmentation_model = MockSegmentationModel()
        segmentation_model.set_image(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        output_path = "output/segmentation_result.jpg"
        segmentation_model.save_result(output_path)
        self.assertTrue(os.path.exists(output_path), "Segmentation result was not saved correctly.")

    def test_detection_model(self):
        class MockDetectionModel(DetectionBaseModel):
            def init_model(self):
                pass

            def set_prompt(self, prompt: str):
                self.prompt = prompt

            def set_image(self, image):
                self.image = image

            def get_result(self):
                pass

            def get_boxes(self) -> Dict[str, List]:
                return {
                    "bboxes": [[50, 50, 200, 200], [300, 300, 400, 400]],
                    "labels": ["person", "bicycle"],
                    "scores": [0.95, 0.88]
                }

        # Instantiate and test the detection model
        detection_model = MockDetectionModel()
        detection_model.set_image(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        output_path = "output/detection_result.jpg"
        detection_model.save_result(output_path)
        self.assertTrue(os.path.exists(output_path), "Detection result was not saved correctly.")


if __name__ == "__main__":
    unittest.main()
