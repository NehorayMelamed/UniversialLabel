import os
import unittest
import cv2
from datetime import datetime
from common.model_name_registry import ModelNameRegistryDetection
from UL.ul_detection import ULDetection
from common.general_parameters import TEST_IMAGE_STREET_DETECTION_PATH


class ULDetectionTest(unittest.TestCase):
    """
    Unit tests for the ULDetection class.
    Each test runs inference on a test image with the detection classes ["car"].
    The results are saved in a directory with a unique, indicative name for each run.
    """

    @classmethod
    def setUpClass(cls):
        cls.image_path = TEST_IMAGE_STREET_DETECTION_PATH  # Path to the test image
        cls.detection_classes = ["car"]  # Detection classes for the models
        cls.output_base_path = "unit_test_results_ul_detection"
        if not os.path.exists(cls.output_base_path):
            os.makedirs(cls.output_base_path)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = os.path.join(cls.output_base_path, f"run_{timestamp}")
        os.makedirs(cls.output_dir, exist_ok=True)

    def _run_ul_detection(self, ul_detection, output_dir):
        """
        Helper function to run the ULDetection class and save results.
        """
        # Load the models
        ul_detection.load_models()

        # Process the image
        nms_results, individual_results = ul_detection.process_image()

        # Save the results
        ul_detection.save_results(individual_results, nms_results, output_dir)

    def test_ul_detection_with_nms(self):
        """
        Test the ULDetection class with Non-Maximum Suppression enabled.
        """
        ul_detection = ULDetection(
            image_input=self.image_path,
            detection_class=self.detection_classes,
            class_priorities={"car": 1},
            model_priorities={
                ModelNameRegistryDetection.WALDO: 1,
                ModelNameRegistryDetection.YOLO_ALFRED: 2,
                ModelNameRegistryDetection.YOLO_WORLD: 3
            },
            use_nms=True,
            sahi_models=[],
            model_names=[
                ModelNameRegistryDetection.YOLO_WORLD,
                ModelNameRegistryDetection.DINO,
                ModelNameRegistryDetection.YOLO_ALFRED,
                ModelNameRegistryDetection.WALDO
            ]
        )

        self._run_ul_detection(ul_detection, self.output_dir)

    def test_ul_detection_without_nms(self):
        """
        Test the ULDetection class without using Non-Maximum Suppression.
        """
        ul_detection = ULDetection(
            image_input=self.image_path,
            detection_class=self.detection_classes,
            class_priorities={"car": 1},
            model_priorities={
                ModelNameRegistryDetection.WALDO: 1,
                ModelNameRegistryDetection.YOLO_ALFRED: 2,
                ModelNameRegistryDetection.YOLO_WORLD: 3
            },
            use_nms=False,
            sahi_models=[],
            model_names=[
                ModelNameRegistryDetection.YOLO_WORLD,
                ModelNameRegistryDetection.DINO,
                ModelNameRegistryDetection.YOLO_ALFRED,
                ModelNameRegistryDetection.WALDO
            ]
        )

        self._run_ul_detection(ul_detection, self.output_dir)


if __name__ == "__main__":
    unittest.main()
