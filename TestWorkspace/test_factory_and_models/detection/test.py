import os
import unittest
import cv2
import numpy as np
from datetime import datetime
from UL.ul_segmentation import ULSegmentation
from common.model_name_registry import ModelNameRegistrySegmentation
from common.general_parameters import TEST_IMAGE_SKY_DETECTION_PATH, TEST_IMAGE_STREET_DETECTION_PATH


class SegmentationModelTest(unittest.TestCase):
    """
    Unit tests for different segmentation models using the UniversalLabelerSegmentation class.
    Each test runs inference on a test image with predefined segmentation classes.
    The results are saved in a directory with a unique, indicative name for each run.
    """

    @classmethod
    def setUpClass(cls):
        cls.sky_image_path = TEST_IMAGE_SKY_DETECTION_PATH  # Path to sky-like test image
        cls.street_image_path = TEST_IMAGE_STREET_DETECTION_PATH  # Path to street-like test image
        cls.segmentation_classes = ["road", "buildings", "pavement", "greenery"]
        cls.output_base_path = "unit_test_results_segmentation"
        if not os.path.exists(cls.output_base_path):
            os.makedirs(cls.output_base_path)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = os.path.join(cls.output_base_path, f"run_{timestamp}")
        os.makedirs(cls.output_dir, exist_ok=True)

    def _run_segmentation(self, ul_segmentation, image_name):
        """Helper function to set up and run the segmentation model, saving the result."""
        # Load models
        ul_segmentation.load_models()

        # Process the image
        formatted_result, individual_results = ul_segmentation.process_image()

        # Save the results
        self._save_results(formatted_result, individual_results, image_name)

    def _save_results(self, formatted_result, individual_results, image_name):
        """Helper function to save model results to the output directory."""
        image_output_dir = os.path.join(self.output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # Save formatted result (final combined result)
        formatted_output_path = os.path.join(image_output_dir, f"{image_name}_formatted_result.txt")
        with open(formatted_output_path, 'w') as f:
            f.write("Final Combined Result:\n")
            f.write(f"Masks: {len(formatted_result['masks'])}\n")
            f.write(f"Labels: {formatted_result['labels']}\n")
            f.write(f"Scores: {formatted_result['scores']}\n")
        print(f"Formatted result for {image_name} saved to {formatted_output_path}")

        # Save individual model results
        for model_name, result in individual_results.items():
            model_output_path = os.path.join(image_output_dir, f"{model_name}_{image_name}_result.txt")
            with open(model_output_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Masks: {len(result['masks'])}\n")
                f.write(f"Labels: {result['labels']}\n")
                f.write(f"Scores: {result['scores']}\n")
            print(f"Individual results for {model_name} on {image_name} saved to {model_output_path}")

    def test_open_earth_map_model_on_sky_image(self):
        """Test OpenEarthMap segmentation model on sky image."""
        ul_segmentation = ULSegmentation(
            image_input=self.sky_image_path,
            segmentation_class=self.segmentation_classes,
            class_priorities={"road": 2, "buildings": 1},
            model_priorities={ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value: 2},
            use_segselector=True,  # Using SegSelector to combine results
            model_names=[ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
        )
        self._run_segmentation(ul_segmentation, "sky_image")

    def test_open_earth_map_model_on_street_image(self):
        """Test OpenEarthMap segmentation model on street image."""
        ul_segmentation = ULSegmentation(
            image_input=self.street_image_path,
            segmentation_class=self.segmentation_classes,
            class_priorities={"road": 2, "buildings": 1},
            model_priorities={ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value: 2},
            use_segselector=True,  # Using SegSelector to combine results
            model_names=[ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
        )
        self._run_segmentation(ul_segmentation, "street_image")


if __name__ == "__main__":
    unittest.main()
