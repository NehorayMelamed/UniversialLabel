import os
import unittest
import cv2
import numpy as np
from datetime import datetime
from UL.ul_segmentation import ULSegmentation
from common.general_parameters import TEST_IMAGE_STREET_DETECTION_PATH, TEST_IMAGE_SKY_DETECTION_PATH
from common.model_name_registry import ModelNameRegistrySegmentation


class ULSegmentationTest(unittest.TestCase):
    """
    Unit tests for the ULSegmentation class.
    Each test runs inference on a test image using multiple segmentation models.
    The results are saved in a directory with a unique, indicative name for each run.
    """

    @classmethod
    def setUpClass(cls):
        cls.sky_image = cv2.imread(TEST_IMAGE_SKY_DETECTION_PATH)  # Random sky-like image for testing
        cls.street_image = cv2.imread(TEST_IMAGE_STREET_DETECTION_PATH)  # Random street-like image for testing
        cls.segmentation_class = ["buildings", "road"]
        cls.output_base_path = "unit_test_results_ul_segmentation"
        if not os.path.exists(cls.output_base_path):
            os.makedirs(cls.output_base_path)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = os.path.join(cls.output_base_path, f"run_{timestamp}")
        os.makedirs(cls.output_dir, exist_ok=True)

    def _run_ul_segmentation(self, ul_segmentation: ULSegmentation, output_dir: str):
        """
        Run the segmentation and save the results.
        """
        # Process the image
        formatted_result, individual_results = ul_segmentation.process_image()

        # Save the results
        ul_segmentation.save_results(formatted_result, output_dir)

    def test_ul_segmentation_with_sky_image(self):
        """
        Test the ULSegmentation class with the sky image.
        """
        ul_segmentation = ULSegmentation(
            image_input=self.sky_image,
            segmentation_class=self.segmentation_class,
            class_priorities={"road": 2, "buildings": 1},
            model_priorities={ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value: 2},
            use_segselector=True,
            model_names=[ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
        )

        # Load the models
        ul_segmentation.load_models()

        # Run and save the results
        output_dir = os.path.join(self.output_dir, "sky_image_results")
        self._run_ul_segmentation(ul_segmentation, output_dir)

    def test_ul_segmentation_with_street_image(self):
        """
        Test the ULSegmentation class with the street image.
        """
        ul_segmentation = ULSegmentation(
            image_input=self.street_image,
            segmentation_class=self.segmentation_class,
            class_priorities={"road": 2, "buildings": 1},
            model_priorities={ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value: 2},
            use_segselector=True,
            model_names=[ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
        )

        # Load the models
        ul_segmentation.load_models()

        # Run and save the results
        output_dir = os.path.join(self.output_dir, "street_image_results")
        self._run_ul_segmentation(ul_segmentation, output_dir)


if __name__ == "__main__":
    unittest.main()
