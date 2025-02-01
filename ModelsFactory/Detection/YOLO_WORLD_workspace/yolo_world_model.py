import cv2

from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from ultralytics import YOLOWorld
from typing import List, Dict, Any
import numpy as np

from common.model_name_registry import ConfigParameters, ModelNameRegistryDetection, PROMPT_MODEL


class YOLOWorld_Model(DetectionBaseModel):
    """
    YOLOWorld_Model handles inference using the YOLO-World API, inheriting from DetectionBaseModel.
    """

    def __init__(self, model_path: str, verbose: bool = False):
        super().__init__(prompt=None)
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.image = None
        self.inference_results = None
        self.model_name = ModelNameRegistryDetection.YOLO_WORLD.value  # Assign model name

    def init_model(self):
        """
        Initialize the YOLO-World model.
        """
        self.model = YOLOWorld(self.model_path)
        if self.verbose:
            print(f"Initialized {self.model_name} model with weights: {self.model_path}")

    def set_prompt(self, prompt: List[str]):
        """
        Set the custom classes or prompt for object detection.

        Args:
            prompt (List[str]): A list of custom class names for object detection.
        """
        # From BaseModel
        self.validate_prompt(prompt)

        self.prompt = prompt
        if self.model:
            self.model.set_classes(self.prompt)

    def set_advanced_parameters(self):
        print(f"{self.__class__.__name__} does not have advanced parameters.")

    def set_image(self, image: np.ndarray):
        """
        Set the input image for the YOLO-World model to process.

        Args:
            image (np.ndarray): The image as a NumPy array.
        """
        self.image = image

    def get_result(self) -> Any:
        """
        Run inference on the image and return the inference result.

        Returns:
            Any: The inference results from the YOLO-World model.
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

        # Extract information from YOLO result object
        boxes = self.inference_results[0].boxes
        for i in range(len(boxes)):
            # Bounding box coordinates (in [x_min, y_min, x_max, y_max] format)
            formatted_result["bboxes"].append(boxes.xyxy[i].tolist())
            # Class label ID
            class_id = int(boxes.cls[i])
            # Convert class ID to class name (using the model's names dictionary)
            formatted_result["labels"].append(self.model.names[class_id])
            # Confidence score
            formatted_result["scores"].append(float(boxes.conf[i]))

        return formatted_result

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that this model uses a free prompt for object detection.

        Returns:
        - str: Notice string.
        """
        return PROMPT_MODEL

# simple test


# Simple usage example of YOLOWorld_Model

if __name__ == "__main__":
    yolo_world = YOLOWorld_Model(model_path=ConfigParameters.YOLO_WORLD_pt.value, verbose=True)

    # Step 2: Initialize the model
    yolo_world.init_model()
    yolo_world.set_image(cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"))
    # Step 3: Set a prompt (classes of interest)
    yolo_world.set_prompt(["car", "bus", "truck", "bike"])
    yolo_world.get_result()
    yolo_world.get_boxes()
    yolo_world.save_result("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/YOLO_WORLD_workspace/result.png")