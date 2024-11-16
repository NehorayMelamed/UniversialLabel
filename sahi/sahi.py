import os
import sys
from typing import Callable, List, Tuple
import cv2
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential  # For applying augmentation sequences to images
from kornia import augmentation as aug
from tqdm import tqdm
from ultralytics import YOLO, YOLOWorld
from PIL import Image



from common.imports import *
sys.path.append(BASE_DIRECTORY_NAME)


from detection_tta_sahi import apply_augmentations, yolo_inference_callback, inverse_transformations, \
    merge_keypoints_to_bbox, draw_bounding_boxes_on_image
from sahi_utils import adjust_bboxes, calculate_slice_bboxes, crop_numpy_batch, crop_tensor, crop_torch_batch, imshow_dgx, pad_to_dimensions, scale_frame

class SahiSlider:
    def __init__(self, model, model_input_dimensions: Tuple[int, int] = (640, 640),
                 slice_dimensions: Tuple[int, int] = (0, 0), detection_conf_threshold: float = 0.5,
                 zoom_factor: float = 1.0, required_overlap_height_ratio: float = 0.2,
                 required_overlap_width_ratio: float = 0.2, transforms: list = None):
        self.model = model
        self.model_input_dimensions = model_input_dimensions
        self.slice_dimensions = slice_dimensions
        self.detection_conf_threshold = detection_conf_threshold
        self.zoom_factor = zoom_factor
        self.required_overlap_height_ratio = required_overlap_height_ratio
        self.required_overlap_width_ratio = required_overlap_width_ratio
        self.transforms = transforms or []
        self.aug_sequence = AugmentationSequential(*self.transforms, data_keys=["input", "keypoints"])

    def set_prompt(self, prompt: List[str]):
        """
        Set the prompt for the underlying model if the model supports custom class prompts.

        Args:
            prompt (List[str]): A list of class labels to detect.
        """
        if hasattr(self.model, 'set_prompt'):
            self.model.set_prompt(prompt)

    def slice_aided_detection_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Performs model inference on image slices, applying transformations and reversing them as needed.

        Args:
            image (np.ndarray): Input image array normalized to [0, 1] of shape (H, W, C).

        Returns:
            np.ndarray: Array of detections with coordinates, classes, and scores.
        """
        predictions = np.empty((0, 6))  # Initialize an empty array for predictions

        # Step 1: Scale the frame to the model's input dimensions
        original_image_dimensions = image.shape[0:2]

        scaled_image = image.copy()
        if self.zoom_factor and self.zoom_factor != 1.0:
            scaled_image, zoom_factor = scale_frame(image, zoom_factor=self.zoom_factor)

        # Step 3: Calculate slice bounding boxes to divide the image into slices
        scaled_image_dimensions = scaled_image.shape[:2]
        scaled_image_height, scaled_image_width = scaled_image_dimensions
        slice_bboxes = [[0, 0, scaled_image_width, scaled_image_height]]

        assert self.slice_dimensions <= (scaled_image_height, scaled_image_width), "Slice dimensions are larger than the scaled image dimensions"

        if self.slice_dimensions != (0, 0) and self.slice_dimensions != (scaled_image_height, scaled_image_width):
            slice_bboxes = calculate_slice_bboxes(scaled_image_dimensions, self.slice_dimensions,
                                                 self.required_overlap_height_ratio, self.required_overlap_width_ratio)

        for idx, (x0, y0, x1, y1) in enumerate(slice_bboxes):
            # crop slice of the image
            slice = scaled_image[y0:y1, x0:x1]  # slice is a numpy array, (slice_H, slice_W, C)
            slice_dimensions = slice.shape[:2]  # (slice_H, slice_W)

            # scale scaled_image_slice for model input dimensions
            slice_scaled_for_model, _ = scale_frame(slice, new_dimensions=self.model_input_dimensions)  # (model_input_H, model_input_W, C)
            slice_scaled_for_model_dimensions = slice_scaled_for_model.shape[:2]  # (model_input_H, model_input_W)

            # switch to tensor
            slice_scaled_for_model_tensor = torch.tensor(slice_scaled_for_model).permute(2, 0, 1).unsqueeze(0).float()  # 1,C,H,W

            # augment the slice
            augmented_slice_scaled_for_model_tensor = apply_augmentations(slice_scaled_for_model_tensor, self.aug_sequence)  # 1,C,H,W

            # apply inference
            augmented_slice_scaled_for_model_predictions = yolo_inference_callback(self.model, augmented_slice_scaled_for_model_tensor,
                                                                                   self.detection_conf_threshold)[0]

            if augmented_slice_scaled_for_model_predictions.shape[0] > 0:
                # inverse transformations
                inversed_keypoints = inverse_transformations(self.aug_sequence, slice_scaled_for_model_tensor,
                                                             augmented_slice_scaled_for_model_predictions)
                slice_scaled_for_model_predictions = merge_keypoints_to_bbox(augmented_slice_scaled_for_model_predictions,
                                                                             inversed_keypoints)

                # rescale slice_predictions from model dimension to slice dimension and offset to match original image
                final_slice_predictions_with_offset = adjust_bboxes(slice_scaled_for_model_predictions,
                                                                    slice_scaled_for_model_dimensions, slice_dimensions,
                                                                    x0, y0)

                # append to predictions
                predictions = np.vstack((predictions, final_slice_predictions_with_offset))

        # rescale all the bounding boxes to the original image size
        predictions = adjust_bboxes(predictions, scaled_image_dimensions, original_image_dimensions)

        return predictions

    def get_bboxes(self, image: np.ndarray) -> dict:
        """
        Get the bounding boxes from the model's inference.

        Args:
            image (np.ndarray): Input image array of shape (H, W, C) and normalized between [0, 1].

        Returns:
            dict: A dictionary containing:
                - "bboxes" (List[List[float]]): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
                - "labels" (List[str]): List of class labels corresponding to each bounding box.
                - "scores" (List[float]): List of confidence scores corresponding to each bounding box.
        """
        predictions = self.slice_aided_detection_inference(image)

        formatted_result = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        for prediction in predictions:
            x_min, y_min, x_max, y_max, cls_id, score = prediction
            formatted_result["bboxes"].append([x_min, y_min, x_max, y_max])
            formatted_result["labels"].append(self.model.names[int(cls_id)])
            formatted_result["scores"].append(float(score))

        return formatted_result

    def save_detection_image_to_disk(self, image: np.ndarray, save_path: str):
        """
        Save the image with bounding boxes and labels to disk.

        Args:
            image (np.ndarray): The input image as a NumPy array. normalized, [0, 1], (H, W, C), float32.
            save_path (str): The path to save the image with bounding boxes and labels.
        """
        predictions = self.slice_aided_detection_inference(image)
        output_image = draw_bounding_boxes_on_image(image, predictions)
        cv2.imwrite(save_path, output_image)
        print(f"Detection result saved to {save_path}")
