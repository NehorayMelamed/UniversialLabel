import os
from typing import List, Dict
import cv2
import numpy as np
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from common.model_name_registry import ModelNameRegistryDetection, PROMPT_MODEL


class DINOXDetection(DetectionBaseModel):
    """
    DINOXDetection class handles object detection tasks using DINO-X via the DDS Cloud API.
    """
    def __init__(self, api_token: str):
        super().__init__()
        self.api_token = api_token
        self.client = None
        self.config = None
        self.model_name = ModelNameRegistryDetection.DINOX_DETECTION
        self.prompts = None
        self.image_url = None
        self.inference_results = None
        self.original_image = None  # Store the original image

    def init_model(self):
        """
        Initialize the DINO-X API client.
        """
        self.config = Config(self.api_token)
        self.client = Client(self.config)

    def set_prompt(self, prompt: List[str]):
        """
        Set the text prompts for DINO-X.

        Args:
            prompt (List[str]): A list of class prompts for detection or segmentation.
        """
        self.validate_prompt(prompt)
        prompts_str = ". ".join(prompt) + "."
        self.prompts = [TextPrompt(text=prompts_str)]

    def set_image(self, image: np.ndarray):
        """
        Upload the input image to the DDS Cloud API.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a NumPy array.")

        self.original_image = image.copy()  # Store the original image
        _, buffer = cv2.imencode('.png', image)
        local_file = "./temp_image.png"

        # Write the image to a temporary local file for upload
        with open(local_file, 'wb') as f:
            f.write(buffer.tobytes())

        # Upload the image and store the URL
        self.image_url = self.client.upload_file(local_file)
        os.remove(local_file)

        if not self.image_url:
            raise ValueError("Failed to upload image to DDS Cloud API.")

    def get_result(self):
        """
        Run inference using DINO-X and return raw results.

        Returns:
            list: Detection objects with bounding boxes and scores.
        """
        if self.image_url is None or self.prompts is None:
            raise ValueError("Image URL or prompts are not set.")

        task = DinoxTask(
            image_url=self.image_url,
            prompts=self.prompts,
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox]
        )
        self.client.run_task(task)
        self.inference_results = task.result.objects
        return self.inference_results

    def get_boxes(self) -> Dict[str, List]:
        """
        Extract bounding boxes, labels, and confidence scores from the inference results.

        Returns:
            dict: Contains "bboxes", "labels", and "scores".
        """
        if not self.inference_results:
            raise ValueError("No inference results found. Please run get_result first.")

        formatted_result = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        for obj in self.inference_results:
            formatted_result["bboxes"].append(obj.bbox)
            formatted_result["labels"].append(obj.category)
            formatted_result["scores"].append(round(obj.score, 3))

        return formatted_result

    def save_result(self, output_path: str):
        """
        Save the detection results as an annotated image.

        Args:
            output_path (str): Path to save the annotated image.
        """
        if self.original_image is None:
            raise ValueError("Original image is not set.")
        if not self.inference_results:
            raise ValueError("Inference results are not available.")

        image_annotated = self.original_image.copy()
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
    # Initialize the DINO-X detection model
    dinox_detection = DINOXDetection(api_token="3fabc3dcc385e7deb0067833ff9da337")
    dinox_detection.init_model()

    # Set prompts and image
    dinox_detection.set_prompt(["wheel", "eye", "helmet", "mouse", "mouth", "vehicle", "steering", "ear", "nose"])
    input_image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/DINO_X_workspace/DINO-X-API/assets/demo.png")

    if input_image is None:
        raise ValueError("Failed to load input image. Check the file path.")

    dinox_detection.set_image(input_image)

    # Run inference and get results
    results = dinox_detection.get_result()
    print(dinox_detection.get_boxes())

    # Save the annotated image
    output_path = "assets/detected_image.jpg"
    dinox_detection.save_result(output_path)
