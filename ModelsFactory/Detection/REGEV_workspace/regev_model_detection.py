from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from common.model_name_registry import ModelNameRegistryDetection, ConfigParameters


class RegevDetectionModel(DetectionBaseModel):
    """
    RegevDetectionModel handles inference using the REGEV detection model, inheriting from DetectionBaseModel.
    """
    CLASS_MAPPING = {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'jeep', 4: 'minibus', 5: 'minivan', 6: 'motorcycle', 7: 'taxi', 8: 'tender', 9: 'truck'}


    @classmethod
    def get_available_classes(cls) -> list:
        return list(cls.CLASS_MAPPING.values())

    def __init__(self, model_path: str, verbose: bool = False):
        super().__init__(prompt=None)
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.inference_results = None
        self.model_name = ModelNameRegistryDetection.YOLO_REGEV.value # Assign model name

    def init_model(self):
        """
        Initialize the YOLO model for REGEV.
        """
        self.model = YOLO(self.model_path)
        if self.verbose:
            print(f"Initialized {self.model_name} model with weights: {self.model_path}")

    def set_prompt(self, prompt: [str]):
        """
        Set the custom classes or prompt for object detection.

        Args:
            prompt (List[str]): A list of custom class names for object detection.
        """
        # From BaseModel
        self.validate_prompt(prompt)

        self.prompt = prompt
        if self.model:
            self.model.classes = self.prompt

    def set_image(self, image: np.ndarray):
        """
        Set the input image for the YOLO model to process.

        Args:
            image (np.ndarray): The image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Expected image as a NumPy array.")
        self.image = image

    def get_result(self):
        """
        Run inference on the image and return the inference result.

        Returns:
            Any: The inference results from the YOLO model.
        """
        if self.image is None:
            raise ValueError("Image not set. Please set an image before calling get_result.")

        self.inference_results = self.model.predict(self.image)
        return self.inference_results

    def get_boxes(self) -> Dict[str, List]:
        """
        Extract bounding boxes, labels, and confidence scores from the inference result.

        Returns:
            Dict[str, List]: A dictionary containing:
                - "bboxes" (List[List[float]]): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
                - "labels" (List[str]): List of class labels corresponding to each bounding box.
                - "scores" (List[float]): List of confidence scores corresponding to each bounding box.
        """
        if self.inference_results is None:
            raise ValueError("No inference results found. Please run get_result() first.")

        formatted_result = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        # Iterate over each detection result to extract bounding boxes, labels, and scores
        for det in self.inference_results[0].boxes:
            bbox = det.xyxy.flatten().tolist()  # Flatten any nested lists
            formatted_result["bboxes"].append(bbox)
            formatted_result["labels"].append(self.CLASS_MAPPING.get(int(det.cls), "Unknown"))
            formatted_result["scores"].append(float(det.conf))

        return formatted_result



# Simple usage example of YOLOWorld_Model

if __name__ == "__main__":
    yolo_world = RegevDetectionModel(model_path=ConfigParameters.YOLO_REGEV_pt.value, verbose=True)

    # Step 2: Initialize the model
    yolo_world.init_model()
    yolo_world.set_image(cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"))
    # Step 3: Set a prompt (classes of interest)
    yolo_world.set_prompt(["car", "bus", "jeep", "bicycle"])
    yolo_world.get_result()
    yolo_world.get_boxes()
    yolo_world.save_result("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/REGEV_workspace/result.png")
