import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Optional, Union
import torch
sys.path.append("git_workspace/sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from common.model_name_registry import ModelNameRegistrySegmentation


class SAM2Segmentation(SegmentationBaseModel):
    """
    SAM2Segmentation integrates the SAM2 model for segmentation tasks.
    """

    def __init__(self, checkpoint_path: str, model_cfg: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.model_name = ModelNameRegistrySegmentation.SAM2.value
        self.predictor = None
        self.image = None
        self.masks = None
        self.labels = []

        # Advanced parameters
        self.mask_threshold = 0.0
        self.max_hole_area = 0.0
        self.max_sprinkle_area = 0.0

    def init_model(self):
        """
        Initialize the SAM2 model.
        """
        self.predictor = SAM2ImagePredictor(
            build_sam2(self.model_cfg, self.checkpoint_path),
            mask_threshold=self.mask_threshold,
            max_hole_area=self.max_hole_area,
            max_sprinkle_area=self.max_sprinkle_area
        )

    def set_advanced_parameters(self, mask_threshold: float = 0.0, max_hole_area: float = 0.0, max_sprinkle_area: float = 0.0):
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
        print(f"Advanced parameters set: mask_threshold={mask_threshold}, max_hole_area={max_hole_area}, max_sprinkle_area={max_sprinkle_area}")

    def set_prompt(self, prompt: str):
        """
        SAM2 does not use text-based prompts. This method is provided for compatibility.

        Args:
            prompt (str): Ignored for SAM2.
        """
        print("SAM2 does not use text prompts. Use predict_on_part for region-specific segmentation.")

    def set_image(self, image: np.ndarray):
        """
        Set the input image for segmentation.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a NumPy array.")
        self.image = image
        self.masks = None  # Reset masks when a new image is set
        self.partial_result = None  # Reset partial results
        self.using_partial_result = False  # Reset partial result flag
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(self.image)

    def get_result(self, boxes: Optional[List[np.ndarray]] = None) -> Dict[str, Union[List[np.ndarray], List[str]]]:
        """
        Perform segmentation on the entire image or specific bounding boxes if provided.

        Args:
            boxes (Optional[List[np.ndarray]]): List of bounding boxes, each in XYXY format.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        if self.image is None:
            raise ValueError("No image set. Use set_image before calling get_result.")

        masks = []
        labels = []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if boxes is not None:
                # Process each bounding box individually
                for box in boxes:
                    print(f"Processing bounding box: {box}")
                    mask, _, _ = self.predictor.predict(box=box)
                    masks.append(mask)
                    labels.append("")  # SAM2 does not provide labels
            else:
                # Perform segmentation on the entire image
                print("Performing segmentation on the entire image.")
                mask, _, _ = self.predictor.predict()
                masks.append(mask)
                labels.append("")  # SAM2 does not provide labels

        self.masks = masks
        self.labels = labels
        return {"masks": masks, "labels": labels}

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
            colored_image[mask > 0.5] = (
                    colored_image[mask > 0.5] * 0.5 + color * 0.5
            ).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_image)
        print(f"Colored segmentation mask saved to {output_path}")

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that SAM2 generates segmentation masks without predefined classes.

        Returns:
            str: Notice string.
        """
        return "SAM2 does not support predefined classes. It generates segmentation masks based on prompts."


if __name__ == "__main__":
    # Initialize the SAM2 model
    sam2_segmentation = SAM2Segmentation(
        checkpoint_path="/home/nehoray/PycharmProjects/UniversaLabeler/common/weights/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    sam2_segmentation.init_model()

    # Set advanced parameters
    sam2_segmentation.set_advanced_parameters(mask_threshold=0.5, max_hole_area=20, max_sprinkle_area=10)

    # Set the image
    image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png")
    sam2_segmentation.set_image(image)

    # Define multiple bounding boxes
    bounding_boxes = [
        np.array([45, 82, 123, 181]),
        np.array([470, 160, 513, 199]),
        np.array([305, 136, 427, 255]),
    ]

    # Perform segmentation for multiple bounding boxes
    results = sam2_segmentation.get_result(boxes=bounding_boxes)

    # Save results
    sam2_segmentation.save_colored_result("output/sam2_results.jpg")

    a = sam2_segmentation.get_bbox_from_masks()
