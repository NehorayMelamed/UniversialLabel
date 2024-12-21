import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_detection_interface import FactoryDetectionInterface
from UL.universal_labeler import UniversalLabeler
from common.model_name_registry import ModelNameRegistryDetection
from sahi.sachi_detection_wrapper import SahiDetectionWrapper
from algorithms.nms_handler import NMSHandler


class ULDetection:
    """
    ULDetection is a universal labeler detection class that processes an image using multiple detection models
    and combines the results, potentially using Non-Maximum Suppression (NMS).
    """

    def __init__(self,
                 image_input: Union[str, np.ndarray],
                 detection_class: List[str],
                 class_priorities: Dict[str, int] = None,
                 model_priorities: Dict[ModelNameRegistryDetection, int] = None,
                 use_nms: bool = True,
                 sahi_models_params: Dict[ModelNameRegistryDetection, Dict] = None,
                 model_names: List[ModelNameRegistryDetection] = None):
        """
        Initialize the ULDetection class.

        Args:
            image_input (Union[str, np.ndarray]): Input image either as a path or a numpy array.
            detection_class (List[str]): List of detection classes or prompts to be used for the detections.
            class_priorities (Dict[str, int]): Dictionary of class priorities for NMS. Defaults to None.
            model_priorities (Dict[ModelNameRegistryDetection, int]): Dictionary of model priorities for NMS. Defaults to None.
            use_nms (bool): Flag to indicate whether to use NMS. Defaults to True.
            sahi_models_params (Dict[ModelNameRegistryDetection, Dict]): Dictionary of SAHI models and their parameters.
            model_names (List[ModelNameRegistryDetection]): List of model names. Defaults to None.
        """
        # Initialize Image
        self.image = self._load_image(image_input)

        # Set detection parameters
        self.detection_class = detection_class
        self.class_priorities = class_priorities if class_priorities else {}
        self.model_priorities = model_priorities if model_priorities else {}
        self.use_nms = use_nms
        self.sahi_models_params = sahi_models_params if sahi_models_params else {}

        # Create models
        self.factory = FactoryDetectionInterface()
        self.models = []
        if model_names is None:
            model_names = self.factory.available_models()
        self.models = self._load_models(model_names)

        # Initialize NMS handler
        self.nms_handler = NMSHandler()
        if self.class_priorities:
            self.nms_handler.set_class_priorities(self.class_priorities)
        if self.model_priorities:
            self.nms_handler.set_model_priorities(self.model_priorities)

    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load an image from a path or directly use the numpy array.
        """
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("Unsupported image input type. Must be a file path or a numpy array.")

    def _load_models(self, model_names: List[ModelNameRegistryDetection]) -> List:
        """
        Load the models from the factory.
        """
        models = []
        for model_name in model_names:
            model = self.factory.create_model(model_name)
            model.init_model()
            model.set_prompt(self.detection_class)
            models.append(model)
        return models

    def load_models(self):
        """
        Load the necessary models, from the FACTORY_INTERFACE_DETECTION.
        """
        for model in self.models:
            model.init_model()

    def process_image(self) -> Dict[str, Dict[str, List]]:
        """
        Process the image with all models and return the formatted result.
        """
        results = {}

        # Process each model
        for model in self.models:
            model_name = model.model_name

            # Set the image for the model
            model.set_image(self.image)

            if model_name in self.sahi_models_params and SahiDetectionWrapper.is_supported(model_name):
                # Use Sahi wrapper if supported
                sahi_wrapper = SahiDetectionWrapper(model)
                # Set custom parameters or use defaults
                sahi_params = self.sahi_models_params.get(model_name, {})
                sahi_wrapper.set_parameters(sahi_params)
                results[model_name] = sahi_wrapper.get_bbox(self.image)
            else:
                if model_name in self.sahi_models_params:
                    print(f"SAHI is not supported for {model_name}. Using regular inference.")

                # Run inference
                model.get_result()

                # Get the bounding boxes from the results
                results[model_name] = model.get_boxes()

        # Format class names to lowercase
        for model_name, result in results.items():
            result['labels'] = [label.lower() for label in result['labels']]

        # Apply NMS if required
        if self.use_nms:
            formatted_result = self.nms_handler.merge_results(results)
        else:
            formatted_result = results

        return formatted_result, results

    def save_results(self, results: Dict[str, Dict[str, List]], nms_results: Dict[str, List], output_dir: str):
        """
        Save detection results for each model and also the final NMS result.

        Args:
            results (Dict[str, Dict[str, List]]): Detection results of each model.
            nms_results (Dict[str, List]): Final NMS results.
            output_dir (str): Directory to save the images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save results for each model
        for model_name, result in results.items():
            output_path = os.path.join(output_dir, f"{model_name}_result.jpg")
            self._save_image_with_boxes(result, output_path)

        # Save final NMS result if available
        if 'bboxes' in nms_results and len(nms_results['bboxes']) > 0:
            output_path = os.path.join(output_dir, "nms_result.jpg")
            self._save_image_with_boxes(nms_results, output_path)
        else:
            print("No NMS results to save.")

    def _save_image_with_boxes(self, result: Dict[str, List], output_path: str):
        """
        Save the image with bounding boxes drawn on it.
        """
        output_image = self.image.copy()
        for bbox, label, score in zip(result['bboxes'], result['labels'], result['scores']):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} ({score:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        cv2.imwrite(output_path, output_image)
        print(f"Detection result saved to {output_path}")


# Example usage
if __name__ == "__main__":
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/tested_image/detection/from_sky.jpeg"
    detection_classes = ["tree", "grass", "car", "person"]
    sahi_model_params = {
        ModelNameRegistryDetection.YOLO_WORLD: {
            'slice_dimensions': (256, 256),
            'detection_conf_threshold': 0.7
        },
        ModelNameRegistryDetection.YOLO_ALFRED: {
            'slice_dimensions': (128, 128),
            'zoom_factor': 1.5
        }
    }
    ul_detection = ULDetection(
        image_input=image_path,
        detection_class=detection_classes,
        class_priorities={},
        model_priorities={},
        use_nms=True,
        # sahi_models_params=sahi_model_params,
        sahi_models_params={},
        model_names=[ModelNameRegistryDetection.YOLO_WORLD, ModelNameRegistryDetection.YOLO_ALFRED]
    )

    # Load the models
    ul_detection.load_models()

    # Process the image
    nms_results, individual_results = ul_detection.process_image()

    # Save the results
    ul_detection.save_results(individual_results, nms_results, "without_sahi")
