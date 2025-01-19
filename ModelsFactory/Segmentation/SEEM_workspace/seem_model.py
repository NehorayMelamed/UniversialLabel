import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Optional, Union
import torch

from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from common.model_name_registry import ModelNameRegistrySegmentation, ConfigParameters

# Import necessary functions from SEEM usage script
from ModelsFactory.Segmentation.SEEM_workspace.git_workspace.seem_usage import load_model, infer_image


class SEEMSegmentation(SegmentationBaseModel):
    """
    SEEMSegmentation integrates the SEEM model for segmentation tasks.
    """

    def __init__(self, checkpoint_path: str, model_cfg: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.model_name = ModelNameRegistrySegmentation.SEEM.value
        self.predictor = None  # Not directly used with SEEM in this context
        self.image = None
        self.masks = None
        self.labels = []
        self.model = None
        self.all_classes = []
        self.colors_list = []

        # Advanced parameters
        self.mask_threshold = 0.0
        self.max_hole_area = 0.0
        self.max_sprinkle_area = 0.0

    def init_model(self):
        """
        Initialize the SEEM model.
        """
        # Load the model using SEEM's load_model function
        self.model, self.all_classes, self.colors_list = load_model(self.model_cfg, self.checkpoint_path)

    def set_advanced_parameters(self, mask_threshold: float = 0.0, max_hole_area: float = 0.0,
                                max_sprinkle_area: float = 0.0):
        """
        Set advanced parameters for mask generation.

        Args:
            mask_threshold (float): Threshold for converting mask logits to binary masks.
            max_hole_area (float): Maximum area of holes to fill in masks.
            max_sprinkle_area (float): Maximum area of small sprinkles to remove in masks.
        """
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        print(
            f"Advanced parameters set: mask_threshold={mask_threshold}, max_hole_area={max_hole_area}, max_sprinkle_area={max_sprinkle_area}")

    def set_prompt(self, prompt):
        # Implementation for setting prompt can be added as needed.
        pass

    def set_image(self, image: np.ndarray):
        """
        Set the input image for segmentation.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if image is None:
            raise ValueError("Invalid image input.")
        self.image = image

    def get_result(self, type="instance") -> Dict[str, Union[List[np.ndarray], List[str]]]:
        """
        Perform segmentation on the entire image.

        Args:
            type (str): The type of segmentation desired. Currently supports only 'instance'.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each detected instance.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call init_model() before get_result().")
        if self.image is None:
            raise RuntimeError("Image not set. Call set_image() before get_result().")

        # Use SEEM's infer_image to get segmentation results
        results, image_np = infer_image(self.model, self.image)

        # Extract instance segmentation results
        instances = results.get('instances', None)
        if instances is not None:
            # Get masks as a list of numpy arrays
            self.masks = [mask.cpu().numpy() for mask in instances.pred_masks]
            # Convert predicted class indices to class names
            pred_classes = instances.pred_classes.cpu().numpy().tolist()
            self.labels = [self.all_classes[c] if c < len(self.all_classes) else "unknown" for c in pred_classes]
        else:
            self.masks = []
            self.labels = []

        # filter empty masks and labels (makss filled with zeros)
        relevant_idx = [idx for idx, mask in enumerate(self.masks) if mask.sum() > 0]
        self.masks = [self.masks[idx] for idx in relevant_idx]
        self.labels = [self.labels[idx] for idx in relevant_idx]

        # reshape masks to match image size if needed
        if len(self.masks) > 0:
            if self.masks[0].shape[0] != self.image.shape[0] or self.masks[0].shape[1] != self.image.shape[1]:
                self.masks = [
                    cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST) for
                    mask in self.masks]

        return {"masks": self.masks, "labels": self.labels}

    def get_masks(self) -> Dict[str, Union[List[np.ndarray], List[str]]]:
        """
        Retrieve binary masks.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        if self.masks is None or not self.labels:
            raise ValueError("Masks or labels are not set. Please run get_result first.")

        return {"masks": self.masks, "labels": self.labels}

    def save_colored_result(self, output_path: str):
        """
        Save segmentation results as a colored overlay using self.masks.

        Args:
            output_path (str): Path to save the result image.
        """
        if self.image is None:
            raise ValueError("Image is not set. Use set_image before saving results.")
        if self.masks is None or len(self.masks) == 0:
            raise ValueError("Masks are not set. Run get_result before saving results.")

        colored_image = self.image.copy()

        for mask in self.masks:
            # Validate mask
            if mask is None:
                print("Warning: Encountered a None mask. Skipping.")
                continue
            if not isinstance(mask, np.ndarray):
                print(f"Warning: Invalid mask type {type(mask)}. Skipping.")
                continue
            if mask.size == 0:
                print("Warning: Empty mask encountered. Skipping.")
                continue

            # Handle multi-channel masks
            if mask.ndim == 3:
                print("Processing multi-channel mask. Combining channels.")
                mask = np.max(mask, axis=0)  # Combine channels by taking the maximum value

            # Generate a random color for the mask
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

            # Overlay the mask on the image
            colored_image[mask > self.mask_threshold] = (
                    colored_image[mask > self.mask_threshold] * 0.5 + color * 0.5
            ).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_image)
        print(f"Colored segmentation mask saved to {output_path}")

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that SEEM generates segmentation masks without predefined classes.

        Returns:
            str: Notice string.
        """
        return "SEEM can support multiple segmentation tasks, including instance, semantic, and panoptic segmentation."


# Example usage
if __name__ == "__main__":
    seem_segmentation = SEEMSegmentation(
        checkpoint_path=ConfigParameters.SEEM_WEIGHTS_FILE_PATH.value,
        model_cfg=ConfigParameters.SEEM_CONFIG_FILE_PATH.value
    )
    seem_segmentation.init_model()

    # Set advanced parameters
    seem_segmentation.set_advanced_parameters(mask_threshold=0.5, max_hole_area=20, max_sprinkle_area=10)

    # Set the image
    image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png")
    seem_segmentation.set_image(image)

    # Perform segmentation
    results = seem_segmentation.get_result(type="instance")
    print(results)

    # Save colored result
    seem_segmentation.save_colored_result("output/seem_results.jpg")
