import os
from abc import abstractmethod
from typing import Dict, List

import cv2

from ModelsFactory.base_model import BaseModel


class DetectionBaseModel(BaseModel):
    """
    DetectionBaseModel extends BaseModel with a specific method to get detection boxes.
    """

    def __init__(self, prompt: str = None):
        super().__init__(prompt)
        self.model = None

    @abstractmethod
    def init_model(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def set_prompt(self, prompt: List[str]):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")


    @abstractmethod
    def set_image(self, image):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def get_result(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def get_boxes(self) -> Dict[str, List]:
        """
        Return the bounding boxes, labels, and scores from the model inference result.

        Returns:
            Dict[str, List]: A dictionary containing:
                - "bboxes" (List[List[float]]): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
                - "labels" (List[str]): List of class labels corresponding to each bounding box.
                - "scores" (List[float]): List of confidence scores corresponding to each bounding box.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    def save_result(self, output_path: str):
        """
        Save the detection result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        if self.image is None:
            raise ValueError("No image set. Please set an image before saving the result.")

        boxes = self.get_boxes()
        output_image = self.image.copy()
        for bbox, label, score in zip(boxes['bboxes'], boxes['labels'], boxes['scores']):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} ({score:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)


        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_image)
        print(f"Detection result saved to {output_path}")
