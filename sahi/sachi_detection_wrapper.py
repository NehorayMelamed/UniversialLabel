from typing import List, Dict
import cv2
import numpy as np
from PIL import Image
from sahi.detection_tta_sahi import draw_bounding_boxes_on_image, slice_aided_detection_inference, yolo_inference_callback
from common.model_name_registry import ModelNameRegistryDetection
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from Factories.factory_detection_interface import FactoryDetectionInterface


class SahiDetectionWrapper:
    """
    A wrapper class to facilitate object detection using the SAHI (Slice Aided Hyper Inference) approach.
    It integrates with supported YOLO models and provides methods for inference, bounding box extraction,
    and saving the detection results.
    """

    SUPPORTED_MODELS = {
        ModelNameRegistryDetection.YOLO_WORLD: ModelNameRegistryDetection.YOLO_WORLD,
        ModelNameRegistryDetection.YOLO_ALFRED: ModelNameRegistryDetection.YOLO_ALFRED,
    }

    def __init__(self, model: DetectionBaseModel):
        """
        Initializes the SahiDetectionWrapper with a given model instance.

        Args:
            model (DetectionBaseModel): An object detection model instance that should be one of the supported models.

        Raises:
            ValueError: If the provided model is not supported.
        """
        if model.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model.model_name} not supported. Supported models: {[model.value for model in self.SUPPORTED_MODELS]}")

        self.model = model
        self.parameters = {
            'model_input_dimensions': (640, 640),
            'slice_dimensions': (0, 0),
            'detection_conf_threshold': 0.5,
            'transforms': [],
            'zoom_factor': 1.0,
            'required_overlap_height_ratio': 0.2,
            'required_overlap_width_ratio': 0.2
        }

    @classmethod
    def is_supported(cls, model_name: ModelNameRegistryDetection) -> bool:
        """
        Checks if a given model name is supported by the wrapper.

        Args:
            model_name (ModelNameRegistryDetection): The name of the model to check.

        Returns:
            bool: True if the model is supported, False otherwise.
        """
        return model_name in cls.SUPPORTED_MODELS

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        Returns a list of all supported model names.

        Returns:
            List[str]: List of supported model names.
        """
        return [model.value for model in cls.SUPPORTED_MODELS]

    def set_parameters(self, parameters: Dict[str, any]):
        """
        Sets custom parameters for the detection wrapper.

        Args:
            parameters (Dict[str, any]): A dictionary containing parameter names and their respective values.
        """
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value

    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Performs inference on the given image using the configured model and parameters.

        Args:
            image (np.ndarray): The input image as a NumPy array, normalized to [0, 1].

        Returns:
            np.ndarray: Array of detections with coordinates, class labels, and confidence scores.

        Raises:
            ValueError: If the model has not been properly loaded.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please provide a valid model instance.")

        predictions = slice_aided_detection_inference(
            image=image,
            model=self.model,
            model_input_dimensions=self.parameters['model_input_dimensions'],
            slice_dimensions=self.parameters['slice_dimensions'],
            detection_conf_threshold=self.parameters['detection_conf_threshold'],
            inference_callback=yolo_inference_callback,
            transforms=self.parameters['transforms'],
            zoom_factor=self.parameters['zoom_factor'],
            required_overlap_height_ratio=self.parameters['required_overlap_height_ratio'],
            required_overlap_width_ratio=self.parameters['required_overlap_width_ratio']
        )
        return predictions

    def get_bbox(self, image: np.ndarray) -> Dict[str, List]:
        """
        Extracts bounding boxes from the image using the model's inference.

        Args:
            image (np.ndarray): Input image as a NumPy array, normalized to [0, 1].

        Returns:
            Dict[str, List]: A dictionary containing bounding boxes, labels, and confidence scores.
        """
        predictions = self.inference(image)
        formatted_result = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        # Determine the source for class labels: CLASS_MAPPING or prompt
        if hasattr(self.model, "CLASS_MAPPING"):
            # Use CLASS_MAPPING if available
            class_mapping = self.model.CLASS_MAPPING
        elif hasattr(self.model, "prompt"):
            # Use the prompt attribute for class labels
            class_mapping = {idx: label for idx, label in enumerate(self.model.prompt)}
        else:
            class_mapping = {}

        # Format the predictions
        for prediction in predictions:
            x_min, y_min, x_max, y_max, cls_id, score = prediction
            label = class_mapping.get(int(cls_id), "Unknown")
            formatted_result["bboxes"].append([x_min, y_min, x_max, y_max])
            formatted_result["labels"].append(label)
            formatted_result["scores"].append(float(score))

        return formatted_result

    def save_result(self, image: np.ndarray, output_path: str):
        """
        Saves the detection results by drawing bounding boxes on the image and saving it to disk.

        Args:
            image (np.ndarray): The input image as a NumPy array, normalized to [0, 1].
            output_path (str): The path where the result image should be saved.
        """
        bbox_result = self.get_bbox(image)
        output_image = draw_bounding_boxes_on_image(image, np.array(bbox_result["bboxes"]))
        output_image = (output_image * 255).astype(np.uint8)
        Image.fromarray(output_image).save(output_path)
        print(f"Detection result saved to {output_path}")

if __name__ == "__main__":
    # Load image as a numpy array
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/sahi/input/img.png"
    # image = Image.open(image_path).convert("RGB")
    # image_np = np.asarray(image) / 255.0
    image_np = cv2.imread(image_path)
    # Instantiate the SahiDetectionWrapper with a pre-loaded model using FactoryDetectionInterface
    factory = FactoryDetectionInterface()
    yolo_model = factory.create_model(ModelNameRegistryDetection.YOLO_WORLD)
    yolo_model.init_model()
    custom_classes = ["bus", "car", "person"]
    yolo_model.set_prompt(custom_classes)
    detection_wrapper = SahiDetectionWrapper(yolo_model)

    # # Set custom parameters
    # custom_parameters = {
    #     'slice_dimensions': (256, 256),
    #     'detection_conf_threshold': 0.7,
    #     'transforms': [aug.RandomHorizontalFlip(p=0.5)]
    # }
    # detection_wrapper.set_parameters(custom_parameters)

    # Run inference and save result
    detection_result = detection_wrapper.get_bbox(image_np)
    print("Detection Result:", detection_result)
    detection_wrapper.save_result(image_np, "detection_result.jpg")
