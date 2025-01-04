import os
from abc import ABC

import cv2
import numpy as np
from typing import Dict, List, Optional, Union
from ultralytics.models.fastsam import FastSAMPredictor
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from common.model_name_registry import ModelNameRegistrySegmentation


class FastSAMSegmentation(SegmentationBaseModel):
    """
    FastSAMSegmentation integrates the FastSAMPredictor for segmentation tasks.
    """

    def __init__(self, model_path: str, conf: float = 0.7, img_size: int = 1024):
        super().__init__()
        self.model_path = model_path
        self.conf = conf
        self.img_size = img_size
        self.model_name = ModelNameRegistrySegmentation.FASTSAM.value
        self.predictor = None
        self.image = None
        self.masks = None
        self.labels = None

    @classmethod
    def get_available_classes(cls) -> list:
        ""


    def set_prompt(self, prompt: str):
        """
        Set any metadata or prompt information (not used in this specific case).
        """
        pass


    def init_model(self):
        """
        Initialize the FastSAM model.
        """
        overrides = dict(
            conf=self.conf,
            task="segment",
            mode="predict",
            model=self.model_path,
            save=False,
            imgsz=self.img_size
        )
        self.predictor = FastSAMPredictor(overrides=overrides)

    def set_image(self, image: np.ndarray):
        """
        Set the input image for segmentation.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a NumPy array.")
        self.image = image
        self.masks = None
        self.labels = None

    def get_result(self, boxes: Optional[List[List[int]]] = None, labels: Optional[List[int]] = None) -> Dict[str, Union[List[np.ndarray], List[int]]]:
        """
        Perform segmentation on the entire image or specific bounding boxes if provided.

        Args:
            boxes (Optional[List[List[int]]]): List of bounding boxes, each in XYXY format.
            labels (Optional[List[int]]): List of class labels corresponding to the bounding boxes.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[int]): List of class labels corresponding to each mask.
        """
        if self.image is None:
            raise ValueError("No image set. Use set_image before calling get_result.")

        results = self.predictor(self.image)
        if boxes is not None:
            prompt_results = self.predictor.prompt(results, bboxes=boxes, labels=labels)
        else:
            prompt_results = results

        self.masks = [r.masks.data.cpu().numpy() for r in prompt_results]
        self.labels = labels if labels is not None else []

        return {"masks": self.masks, "labels": self.labels}

    def save_colored_result(self, output_path: str):
        """
        Save segmentation results as a colored overlay.

        Args:
            output_path (str): Path to save the result image.
        """
        if self.image is None:
            raise ValueError("Image is not set. Use set_image before saving results.")
        if self.masks is None:
            raise ValueError("Masks are not set. Run get_result before saving results.")

        colored_image = self.image.copy()

        for idx, mask in enumerate(self.masks):
            # Validate and preprocess the mask
            if mask.ndim > 2:  # Multi-channel mask
                print("Processing multi-channel mask. Combining channels.")
                mask = np.max(mask, axis=0)  # Combine channels

            if mask.shape != self.image.shape[:2]:  # Resize mask to match image size
                mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Ensure the mask is binary
            binary_mask = mask > 0

            # Generate a random color
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

            # Overlay the mask on the image
            colored_image[binary_mask] = (
                    colored_image[binary_mask] * 0.5 + color * 0.5
            ).astype(np.uint8)

            print(f"Processed mask {idx}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_image)
        print(f"Colored segmentation mask saved to {output_path}")

    def get_masks(self) -> Dict[str, Union[List[np.ndarray], List[int]]]:
        """
        Retrieve binary masks.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[int]): List of class labels corresponding to each mask.
        """
        if self.masks is None or not self.labels:
            raise ValueError("Masks or labels are not set. Please run get_result first.")

        return {"masks": self.masks, "labels": self.labels}

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that FastSAM supports flexible prompts.

        Returns:
            str: Notice string.
        """
        return "FastSAM supports segmentation with bounding boxes, points, and textual prompts."


if __name__ == "__main__":
    # Initialize the FastSAM model
    fastsam_segmentation = FastSAMSegmentation(
        model_path="../../../common/weights/FastSAM-x.pt",
        conf=0.7,
        img_size=1024
    )
    fastsam_segmentation.init_model()

    # Set the image
    image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png")
    fastsam_segmentation.set_image(image)

    # Define multiple bounding boxes and labels
    bounding_boxes = [
        [45, 82, 123, 181],
        [470, 160, 513, 199],
        [305, 136, 427, 255]
    ]
    labels = [1, 2, 3]

    # Perform segmentation for specific bounding boxes
    results = fastsam_segmentation.get_result(boxes=bounding_boxes, labels=labels)

    # Save results
    fastsam_segmentation.save_colored_result("output/fastsam_results.jpg")
