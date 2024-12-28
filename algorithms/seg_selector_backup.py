import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_segmentation_interface import FactorySegmentationInterface
from common.model_name_registry import ModelNameRegistrySegmentation

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

    def merge_results(self, results: Dict[str, Dict[str, List[np.ndarray]]], image_shape: tuple) -> Dict[
        str, Dict[str, List]]:
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
            resized_masks = []
            for mask in result['masks']:
                # Validate mask
                if mask is None:
                    print(f"Warning: Mask is None for model {model_name}. Skipping.")
                    continue
                if not isinstance(mask, np.ndarray):
                    print(f"Warning: Invalid mask type {type(mask)} for model {model_name}. Skipping.")
                    continue
                if mask.size == 0:
                    print(f"Warning: Empty mask for model {model_name}. Skipping.")
                    continue

                # Handle multi-channel masks (e.g., [n, h, w])
                if mask.ndim == 3:
                    print(f"Processing multi-channel mask for model {model_name}. Keeping all channels.")
                    # Resize each channel if needed
                    mask_channels = []
                    for channel in mask:
                        if channel.shape[:2] != image_shape[:2]:
                            print(
                                f"Resizing channel from {channel.shape[:2]} to {image_shape[:2]} for model {model_name}.")
                            channel = cv2.resize(channel, (image_shape[1], image_shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                        mask_channels.append(channel)
                    # Combine resized channels back into multi-channel mask
                    mask = np.stack(mask_channels, axis=0)

                # Ensure mask is resized to the correct image shape for single-channel masks
                elif mask.ndim == 2 and mask.shape[:2] != image_shape[:2]:
                    print(f"Resizing mask from {mask.shape[:2]} to {image_shape[:2]} for model {model_name}.")
                    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

                resized_masks.append(mask)

            # Add formatted results for the model
            formatted_result[model_name] = {
                'masks': resized_masks,
                'labels': result.get('labels', []),
                'scores': [1.0] * len(resized_masks)  # Placeholder scores
            }

        return formatted_result
