import os
from PIL import Image
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from typing import List, Dict
import numpy as np
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from common.model_name_registry import PROMPT_MODEL, ModelNameRegistryDetection, ConfigParameters


class DetrDetectionModel(DetectionBaseModel):
    """
    DetrDetectionModel handles object detection using the DETR API, inheriting from DetectionBaseModel.
    """

    def __init__(self, processor_path: str, model_path: str, threshold: float = 0.9):
        super().__init__()
        self.processor_path = processor_path
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.threshold = threshold
        self.model_name = ModelNameRegistryDetection.DETR.value # Replace with the appropriate registry key


    def init_model(self):
        """
        Initialize the DETR model and processor.
        """
        self.processor = DetrImageProcessor.from_pretrained(self.processor_path)
        self.model = DetrForObjectDetection.from_pretrained(self.model_path)

    def set_advanced_parameters(self):
        print(f"{self.__class__.__name__} does not have advanced parameters.")

    def set_prompt(self, prompt: List[str]):
        """
        DETR does not use prompts. Method is overridden to maintain compatibility.
        """
        print("DETR does not support prompts. This method is ignored.")

    def set_image(self, image: np.ndarray):
        """
        Set the input image for DETR.

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
            dict: Raw inference results from the DETR model.
        """
        if self.image is None:
            raise ValueError("Image is not set. Please set an image before calling get_result.")

        image_pil = Image.fromarray(self.image[:, :, ::-1])  # Convert to PIL image in RGB format
        inputs = self.processor(images=image_pil, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]])
        self.inference_results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
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

        for score, label, box in zip(
            self.inference_results["scores"],
            self.inference_results["labels"],
            self.inference_results["boxes"]
        ):
            formatted_result["bboxes"].append([round(i, 2) for i in box.tolist()])
            formatted_result["labels"].append(self.model.config.id2label[label.item()])
            formatted_result["scores"].append(round(score.item(), 3))

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
        DETR does not have predefined classes, uses a "free prompt".

        Returns:
            str: "free prompt".
        """
        return PROMPT_MODEL



if __name__ == '__main__':

    # Initialize the DETR detection model
    detr_model = DetrDetectionModel(processor_path=ConfigParameters.DERT_MODEL.value,
                                    model_path=ConfigParameters.DERT_PROCESSOR.value  )
    detr_model.init_model()

    # Load and set the image
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/airport.jpg"
    image = cv2.imread(image_path)
    detr_model.set_image(image)

    # Run inference
    detr_model.get_result()

    # Retrieve bounding boxes
    boxes = detr_model.get_boxes()
    print(boxes)

    # Save annotated result
    output_path = "/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/facebook_detr_workspace/output.jpg"
    detr_model.save_result(output_path)
