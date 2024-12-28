import os
from typing import Optional, List

import torch

from ModelsFactory.base_model import BaseModel
from abc import abstractmethod
import numpy as np
import cv2


class SegmentationBaseModel(BaseModel):
    """
    Abstract base class for segmentation models.
    Inherits from BaseModel and adds the get_masks method.
    """

    def __init__(self, prompt: str = None):
        super().__init__(prompt)
        self.model_name = None
        self.image = None

    @abstractmethod
    def init_model(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def set_prompt(self, prompt: str):
        self.prompt = prompt

    @abstractmethod
    def set_image(self, image):
        """
        Set the input image for the model.
        Args:
            image: Input image to be processed by the model.
        """
        self.image = image

    @abstractmethod
    def get_result(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def get_masks(self) -> dict:
        """
        Retrieve the segmentation masks from the model's output.
        This should be implemented by all segmentation models.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def save_colored_result(self, output_path: str):
        """
        Save the colored segmentation mask result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    def save_result(self, output_path: str):
        """
        Save the segmentation mask result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        if self.image is None:
            raise ValueError("No image set. Please set an image before saving the result.")

        masks_data = self.get_masks()
        masks = masks_data.get("masks", [])
        labels = masks_data.get("labels", [])

        if not masks:
            raise ValueError("No masks available to save.")

        # Combine all masks into a single image for visualization
        combined_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)  # Use spatial dimensions of the image
        for idx, mask in enumerate(masks):
            if mask.shape != combined_mask.shape:
                mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            combined_mask[mask > 0] = (idx + 1) * 40  # Assign different intensity for each class to improve visualization

        result_image = cv2.merge([combined_mask, combined_mask, combined_mask])  # Convert to 3-channel image for saving
        cv2.imwrite(output_path, result_image)
        print(f"Segmentation mask saved to {output_path}")

    def format_segmentation_result(self, result: np.ndarray, class_mapping: dict) -> dict:
        """
        Format the segmentation result to ensure consistency between different segmentation models.
        This method ensures the result is formatted in the same way regardless of whether the segmentation is semantic or instance.

        Args:
            result (np.ndarray): The raw output of the segmentation model.
            class_mapping (dict): A dictionary mapping class indices to class labels.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        # Convert raw model output to class predictions
        if result.ndim == 3:
            # Assuming the result is of shape [num_classes, height, width]
            predicted_classes = np.argmax(result, axis=0)
        else:
            predicted_classes = result

        masks = []
        labels = []
        unique_classes = np.unique(predicted_classes)
        for class_index in unique_classes:
            if class_index in class_mapping:
                mask = (predicted_classes == class_index).astype(np.uint8)
                masks.append(mask)
                labels.append(class_mapping[class_index])

        return {"masks": masks, "labels": labels}

    def get_bbox_from_masks(self, margin: int = 10) -> List[List[int]]:
        """
        Calculate bounding boxes from masks with an optional margin.

        Args:
            margin (int): Margin to add around the bounding boxes. Defaults to 10.

        Returns:
            List[List[int]]: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
        """
        if self.image is None:
            raise ValueError("Image is not set. Use set_image before calling this method.")

        # Handle image conversion based on its type
        if isinstance(self.image, torch.Tensor):
            # Convert torch.Tensor to NumPy array
            image_numpy = self.image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            image_numpy = image_numpy.astype(np.uint8)  # Convert to uint8 for OpenCV
        elif isinstance(self.image, np.ndarray):
            # If already a NumPy array, ensure dtype is uint8
            image_numpy = self.image
            if image_numpy.dtype != np.uint8:
                image_numpy = (image_numpy * 255).astype(np.uint8)
        else:
            raise ValueError("Unsupported image type. Expected torch.Tensor or np.ndarray.")

        masks_data = self.get_masks()
        masks = masks_data.get("masks", [])
        if not masks:
            raise ValueError("No masks available to calculate bounding boxes.")

        bboxes = []
        output_image = image_numpy.copy()  # Create a copy of the NumPy array

        for idx, mask in enumerate(masks):
            # Ensure the mask is single-channel
            if mask.ndim == 3:  # If more than one channel exists
                mask = np.any(mask > 0, axis=0)

            # Find all non-zero pixels in the mask
            coords = np.column_stack(np.where(mask > 0))
            if coords.size == 0:  # If the mask is empty
                continue

            # Calculate bounding box boundaries
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            # Add margin
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image_numpy.shape[1], x_max + margin)
            y_max = min(image_numpy.shape[0], y_max + margin)

            # Save the bounding box
            bboxes.append([x_min, y_min, x_max, y_max])

            # Draw the bounding box on the image
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                output_image,
                f"Mask {idx}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return bboxes
