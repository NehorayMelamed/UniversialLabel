import os
import cv2
import numpy as np
from typing import Dict, List, Union


class SegSelector:
    """
    A class responsible for combining segmentation results from multiple models
    based on class and model priorities.
    """

    def __init__(self):
        self.class_priorities = {}
        self.model_priorities = {}

    def set_class_priorities(self, priorities: Dict[str, int]):
        """
        Set priorities for classes.
        Args:
            priorities (Dict[str, int]): Dictionary of class priorities. Higher value indicates higher priority.
        """
        self.class_priorities = priorities

    def set_model_priorities(self, priorities: Dict[str, int]):
        """
        Set priorities for models.
        Args:
            priorities (Dict[str, int]): Dictionary of model priorities. Higher value indicates higher priority.
        """
        self.model_priorities = priorities

    def merge_results(self, results: Dict[str, Dict[str, List[np.ndarray]]], image_shape: tuple) -> Dict[str, Dict[str, List]]:
        """
        Merge segmentation results from multiple models based on class and model priorities.

        Args:
            results (Dict[str, Dict[str, List[np.ndarray]]]): Dictionary containing segmentation results for each model.
            image_shape (tuple): Shape of the input image to ensure masks match the correct size.

        Returns:
            Dict[str, Dict[str, List]]: A dictionary containing the final formatted result similar to detection.
        """
        formatted_result = {}
        for model_name, result in results.items():
            # Resize masks to match the input image shape
            resized_masks = [
                cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                if mask.shape[:2] != image_shape[:2] else mask
                for mask in result['masks']
            ]

            formatted_result[model_name] = {
                'masks': resized_masks,
                'labels': result['labels'],
                'scores': [1.0] * len(resized_masks)  # Placeholder scores
            }

        return formatted_result
