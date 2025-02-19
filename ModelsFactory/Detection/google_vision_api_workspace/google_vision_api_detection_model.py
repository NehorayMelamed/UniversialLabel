import os
from google.cloud import vision
from typing import List, Dict
import cv2
import numpy as np
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from common.model_name_registry import ModelNameRegistryDetection, ConfigParameters, PROMPT_MODEL


class GoogleVisionDetectionModel(DetectionBaseModel):
    """
    GoogleVisionDetectionModel handles object detection using the Google Vision API, inheriting from DetectionBaseModel.
    """
    def set_prompt(self, prompt: List[str]):
        """
        google vision does not support prompts. Method is overridden to maintain compatibility.
        """
        print("google vision does not support prompts. This method is ignored.")

    def __init__(self, credential_path: str):
        super().__init__()
        self.credential_path = credential_path
        self.client = None
        self.image = None
        self.model_name = ModelNameRegistryDetection.GOOGLE_VISION.value

    def set_advanced_parameters(self):
        print(f"{self.__class__.__name__} does not have advanced parameters.")

    def init_model(self):
        """
        Initialize the Google Vision API client.
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credential_path
        self.client = vision.ImageAnnotatorClient()

    def set_image(self, image: np.ndarray):
        """
        Set the input image for Google Vision.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a NumPy array.")
        self.image = image

    def get_result(self):
        """
        Run inference on the input image and return raw results.

        Returns:
            list: Detected objects from the Google Vision API.
        """
        if self.image is None:
            raise ValueError("Image is not set. Please set an image before calling get_result.")

        _, buffer = cv2.imencode('.jpg', self.image)
        image_content = buffer.tobytes()
        image = vision.Image(content=image_content)

        response = self.client.object_localization(image=image)
        self.inference_results = response.localized_object_annotations
        return self.inference_results

    def get_boxes(self) -> Dict[str, List]:
        """
        Extract bounding boxes, labels, and confidence scores from the inference result.

        Returns:
            dict: Contains "bboxes", "labels", and "scores".
        """
        if self.inference_results is None:
            raise ValueError("No inference results found. Please run get_result first.")

        formatted_result = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        for obj in self.inference_results:
            # Extract normalized bounding box vertices
            vertices = obj.bounding_poly.normalized_vertices

            # Ensure vertices are valid (must have exactly 4 points)
            if len(vertices) != 4:
                print(f"Skipping invalid bbox: {vertices}")
                continue

            # Convert normalized vertices to [x_min, y_min, x_max, y_max] format
            x_min = min(v.x for v in vertices) * self.image.shape[1]
            y_min = min(v.y for v in vertices) * self.image.shape[0]
            x_max = max(v.x for v in vertices) * self.image.shape[1]
            y_max = max(v.y for v in vertices) * self.image.shape[0]

            formatted_result["bboxes"].append([x_min, y_min, x_max, y_max])
            formatted_result["labels"].append(obj.name)
            formatted_result["scores"].append(round(obj.score, 3))

        return formatted_result

    def save_result(self, output_path: str):
        """
        Save the detection results as an annotated image.

        Args:
            output_path (str): Path to save the annotated image.
        """
        if self.image is None or self.inference_results is None:
            raise ValueError("Image or inference results are not set.")

        image_annotated = self.image.copy()
        boxes = self.get_boxes()

        for bbox, label, score in zip(boxes["bboxes"], boxes["labels"], boxes["scores"]):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image_annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                image_annotated,
                f"{label} ({score:.2f})",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_annotated)

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that this model uses a free prompt for object detection.

        Returns:
        - str: Notice string.
        """
        return PROMPT_MODEL

if __name__ == '__main__':
    # Initialize the Google Vision detection model
    google_vision_model = GoogleVisionDetectionModel(
        credential_path=ConfigParameters.GOOGLE_VISION_KEY_API.value
    )
    google_vision_model.init_model()

    # Load and set the image
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/airport.jpg"
    image = cv2.imread(image_path)
    google_vision_model.set_image(image)

    # Run inference
    google_vision_model.get_result()

    # Retrieve bounding boxes
    boxes = google_vision_model.get_boxes()
    print(boxes)

    # Save annotated result
    output_path = "/home/nehoray/PycharmProjects/UniversaLabeler/new_models_workspace/google_vision/output.jpg"
    google_vision_model.save_result(output_path)
