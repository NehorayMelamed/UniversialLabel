import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_detection_interface import FactoryDetectionInterface
from UL.universal_labeler import UniversalLabeler
from common.model_name_registry import ModelNameRegistryDetection, MOST_CONFIDENCE
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
                 model_names: List[ModelNameRegistryDetection] = None,
                 filter_unwanted_classes: bool = True,
                 trex_input_class_bbox: Dict[str, Union[str, List[int], ModelNameRegistryDetection]] = None):
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
            filter_unwanted_classes (bool): Flag to enable or disable filtering of unwanted classes. Defaults to True.
            trex_input_class_bbox (Dict[str, Union[str, List[int], ModelNameRegistryDetection]]): Configuration for TREX input.
        """
        # Initialize Image
        self.image = self._load_image(image_input)

        # Set detection parameters
        self.detection_class = detection_class
        self.class_priorities = class_priorities if class_priorities else {}
        self.model_priorities = model_priorities if model_priorities else {}
        self.use_nms = use_nms
        self.sahi_models_params = sahi_models_params if sahi_models_params else {}
        self.filter_unwanted_classes = filter_unwanted_classes
        self.trex_input_class_bbox = trex_input_class_bbox

        # Create models
        self.factory = FactoryDetectionInterface()
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
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("Unsupported image input type. Must be a file path or a numpy array.")

    def _load_models(self, model_names: List[ModelNameRegistryDetection]) -> List:
        models = []
        for model_name in model_names:
            model = self.factory.create_model(model_name)
            model.init_model()
            if model_name != ModelNameRegistryDetection.TREX2:
                model.set_prompt(self.detection_class)
            models.append(model)
        return models

    def process_image(self) -> Dict[str, Dict[str, List]]:
        results = {}

        # Process each model
        for model in self.models:
            model_name = model.model_name
            model.set_image(self.image)

            if model_name in self.sahi_models_params and SahiDetectionWrapper.is_supported(model_name):
                sahi_wrapper = SahiDetectionWrapper(model)
                sahi_params = self.sahi_models_params.get(model_name, {})
                sahi_wrapper.set_parameters(sahi_params)
                results[model_name] = sahi_wrapper.get_bbox(self.image)
            elif model_name == ModelNameRegistryDetection.TREX2:
                trex_prompts = self._build_trex_prompts(results)
                model.set_prompt(trex_prompts)
                model.get_result()
                results[model_name] = model.get_boxes()
            else:
                model.get_result()
                results[model_name] = model.get_boxes()

        # Format class names to lowercase
        for model_name, result in results.items():
            result['labels'] = [label.lower() for label in result['labels']]

        # Apply NMS if required
        if self.use_nms:
            formatted_result = self.nms_handler.merge_results(results)
        else:
            formatted_result = results

        if self.filter_unwanted_classes:
            formatted_result = self.filter_classes(formatted_result)
            results = {model_name: self.filter_classes(model_result) for model_name, model_result in results.items()}

        return formatted_result, results

    def _build_trex_prompts(self, results: Dict[str, Dict[str, List]]) -> Dict[str, List[List[int]]]:
        trex_prompts = {}
        for class_name, config in self.trex_input_class_bbox.items():
            if config == MOST_CONFIDENCE:
                highest_score = 0
                best_bbox = None
                for model_result in results.values():
                    for bbox, label, score in zip(model_result['bboxes'], model_result['labels'], model_result['scores']):
                        if label == class_name and score > highest_score:
                            highest_score = score
                            best_bbox = bbox
                if best_bbox:
                    trex_prompts[class_name] = [best_bbox]
            elif isinstance(config, ModelNameRegistryDetection):
                model_result = results.get(config.value)
                if model_result:
                    highest_score_idx = np.argmax(model_result['scores'])
                    trex_prompts[class_name] = [model_result['bboxes'][highest_score_idx]]
            elif isinstance(config, list) and len(config) == 4:
                trex_prompts[class_name] = [config]
            else:
                print(f"Invalid TREX configuration for class {class_name}: {config}")
        return trex_prompts

    def filter_classes(self, results: Dict[str, List]) -> Dict[str, List]:
        filtered_results = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        for bbox, label, score in zip(results.get("bboxes", []), results.get("labels", []), results.get("scores", [])):
            if label in self.detection_class:
                filtered_results["bboxes"].append(bbox)
                filtered_results["labels"].append(label)
                filtered_results["scores"].append(score)

        return filtered_results

    def save_results(self, results: Dict[str, Dict[str, List]], nms_results: Dict[str, List], output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for model_name, result in results.items():
            output_path = os.path.join(output_dir, f"{model_name}_result.jpg")
            self._save_image_with_boxes(result, output_path)

        if 'bboxes' in nms_results and len(nms_results['bboxes']) > 0:
            output_path = os.path.join(output_dir, "nms_result.jpg")
            self._save_image_with_boxes(nms_results, output_path)
        else:
            print("No NMS results to save.")

    def _save_image_with_boxes(self, result: Dict[str, List], output_path: str):
        output_image = self.image.copy()
        for bbox, label, score in zip(result['bboxes'], result['labels'], result['scores']):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} ({score:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        cv2.imwrite(output_path, output_image)
        print(f"Detection result saved to {output_path}")

if __name__ == "__main__":
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"
    detection_classes = ["car", "bus"]
    # trex_input_class_bbox = {
    #     "car": MOST_CONFIDENCE,
    #     "bus": MOST_CONFIDENCE
    # }
    trex_input_class_bbox = {
        "bus": [313, 141, 406,249],
        "car":[471, 162, 514, 197]
    }

    sahi_model_params = {
        ModelNameRegistryDetection.YOLO_WORLD: {
            'slice_dimensions': (256, 256),
            'detection_conf_threshold': 0.7
        }
    }

    ul_detection = ULDetection(
        image_input=image_path,
        detection_class=detection_classes,
        # class_priorities={"car": 2, "person": 1},
        model_priorities={ModelNameRegistryDetection.YOLO_WORLD: 2, ModelNameRegistryDetection.TREX2: 0},
        use_nms=True,
        # sahi_models_params=sahi_model_params,
        model_names=[ModelNameRegistryDetection.YOLO_WORLD, ModelNameRegistryDetection.TREX2],
        filter_unwanted_classes=True,
        trex_input_class_bbox=trex_input_class_bbox
    )

    # Process the image
    nms_results, individual_results = ul_detection.process_image()

    # Save the results
    output_directory = "./output/trex_with_box"
    ul_detection.save_results(individual_results, nms_results, output_directory)
