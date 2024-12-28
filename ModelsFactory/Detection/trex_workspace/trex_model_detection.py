import os
import cv2
import numpy as np
from typing import List, Dict
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from common.model_name_registry import ModelNameRegistryDetection, PROMPT_MODEL_BASE_INFERENCE_BBOX
from ModelsFactory.Detection.trex_workspace.model_wrapper import TRex2APIWrapper


class TRex2DetectionModel(DetectionBaseModel):
    """
    TRex2DetectionModel handles object detection using the T-Rex2 API, inheriting from DetectionBaseModel.
    """

    def __init__(self, token: str, box_threshold: float = 0.3):
        super().__init__()
        self.token = token
        self.box_threshold = box_threshold
        self.trex2_api = None
        self.image = None
        self.prompts = None
        self.inference_results = None
        self.model_name = ModelNameRegistryDetection.TREX2.value
        self.class_to_id = {}  # Mapping of class names to numerical IDs
        self.id_to_class = {}  # Reverse mapping for decoding results

    def init_model(self):
        """
        Initialize the T-Rex2 API.
        """
        self.trex2_api = TRex2APIWrapper(self.token)

    def set_prompt(self, categories_with_boxes: Dict[str, List[List[int]]]):
        """
        Set prompts for T-Rex2 using a dictionary of category names and bounding boxes.

        Args:
            categories_with_boxes (Dict[str, List[List[int]]]): A dictionary where the keys are category names
            and the values are lists of bounding boxes in the format [xmin, ymin, xmax, ymax].
        """
        if not isinstance(categories_with_boxes, dict):
            raise ValueError("Prompts must be a dictionary with category names as keys and bounding boxes as values.")

        # Map category names to numerical IDs
        self.class_to_id = {name: idx + 1 for idx, name in enumerate(categories_with_boxes.keys())}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        # Convert category names to numerical IDs for the API
        self.prompts = [
            {
                "prompt_image": self.image,
                "type": "rect",
                "prompts": [{"category_id": self.class_to_id[name], "rects": rects}
                            for name, rects in categories_with_boxes.items()],
            }
        ]

    def set_image(self, image: np.ndarray):
        """
        Set the input image for T-Rex2, saving it as a temporary file.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a NumPy array.")

        temp_path = "./temp_image.jpg"
        cv2.imwrite(temp_path, image)
        self.image = temp_path

    def get_result(self):
        """
        Run inference on the input image and return filtered results.

        Returns:
            List[Dict]: Filtered inference results from T-Rex2 API.
        """
        if not self.image or not self.prompts:
            raise ValueError("Image or prompts are not set. Please set both before running inference.")

        results = self.trex2_api.interactve_inference(self.prompts)
        # Filter results based on the box threshold
        filtered_results = []
        for result in results:
            scores = np.array(result["scores"])
            labels = np.array(result["labels"])
            boxes = np.array(result["boxes"])
            filter_mask = scores > self.box_threshold
            filtered_result = {
                "scores": scores[filter_mask].tolist(),
                "labels": [self.id_to_class[label] for label in labels[filter_mask]],  # Convert back to class names
                "boxes": boxes[filter_mask].tolist(),
            }
            filtered_results.append(filtered_result)
        self.inference_results = filtered_results
        return self.inference_results

    def get_boxes(self) -> Dict[str, List]:
        """
        Extract bounding boxes, labels, and confidence scores from the inference result.

        Returns:
            dict: Contains "bboxes", "labels", and "scores".
        """
        if self.inference_results is None:
            raise ValueError("No inference results found. Please run get_result first.")

        combined_results = {
            "bboxes": [],
            "labels": [],
            "scores": [],
        }

        for result in self.inference_results:
            combined_results["bboxes"].extend(result["boxes"])
            combined_results["labels"].extend(result["labels"])
            combined_results["scores"].extend(result["scores"])

        return combined_results

    def save_result(self, output_path: str):
        """
        Save the detection results as an annotated image.

        Args:
            output_path (str): Path to save the annotated image.
        """
        if self.image is None or self.inference_results is None:
            raise ValueError("Image or inference results are not set.")

        # Load the image
        image_annotated = cv2.imread(self.image)

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
        Return available classes or categories.

        Returns:
            str: "T-Rex2 API supports custom prompts."
        """
        return PROMPT_MODEL_BASE_INFERENCE_BBOX
        # pass


if __name__ == '__main__':
    # Example Usage
    model = TRex2DetectionModel(token="9d01f376fbf2bb13123c85a82da8e154")
    model.init_model()

    # Set the image
    input_image = cv2.imread("/home/nehoray/PycharmProjects/tests_wotkspace/T-Rex/assets/trex2_api_examples/interactive1.jpeg")
    model.set_image(input_image)

    # Define prompts with category names and bounding boxes
    categories_with_boxes = {
        "car": [[468.04302978515625, 159.61497497558594, 514.6063842773438, 199.15711975097656]],  # Category: car
        "person": [[150, 150, 250, 250]],  # Category: person
    }
    model.set_prompt(categories_with_boxes)

    # Run inference
    results = model.get_result()
    print("Results:", results)

    # Save annotated image
    output_path = "output_directory/annotated_image.jpg"
    model.save_result(output_path)
