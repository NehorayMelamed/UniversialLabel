import sys
sys.path.append("../")
import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_detection_interface import FactoryDetectionInterface
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
                 model_priorities: Dict[str, int] = None,
                 use_nms: bool = True,
                 sahi_models_params: Dict[str, Dict] = None,
                 model_names: List[str] = None,
                 filter_unwanted_classes: bool = True,
                 trex_input_class_bbox: Dict[str, Union[str, List[int], str]] = None):
        """
        Initialize the ULDetection class.

        Args:
            image_input (Union[str, np.ndarray]): Input image either as a path or a numpy array.
            detection_class (List[str]): List of detection classes or prompts to be used for the detections.
            class_priorities (Dict[str, int]): Dictionary of class priorities for NMS. Defaults to None.
            model_priorities (Dict[str, int]): Dictionary of model priorities for NMS. Defaults to None.
            use_nms (bool): Flag to indicate whether to use NMS. Defaults to True.
            sahi_models_params (Dict[str, Dict]): Dictionary of SAHI models and their parameters.
            model_names (List[str]): List of model names as strings. Defaults to None.
            filter_unwanted_classes (bool): Flag to enable or disable filtering of unwanted classes. Defaults to True.
            trex_input_class_bbox (Dict[str, Union[str, List[int], str]]): Configuration for TREX input.
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
            model_names = [model.value for model in ModelNameRegistryDetection]

        # if the user pass the trex args but not to the models
        if ModelNameRegistryDetection.TREX2.value not in model_names and trex_input_class_bbox is not None:
            model_names.append(ModelNameRegistryDetection.TREX2.value)

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

    def _load_models(self, model_names: List[str]) -> List:
        models = []
        for model_name in model_names:
            model = self.factory.create_model(model_name)
            model.init_model()
            if model_name != ModelNameRegistryDetection.TREX2.value:
                model.set_prompt(self.detection_class)
            models.append(model)
        return models

    def set_image(self, image_input: Union[str, np.ndarray]):
        """
        Set a new image for processing.

        Args:
            image_input (Union[str, np.ndarray]): Input image either as a path or a numpy array.
        """
        self.image = self._load_image(image_input)

    def set_sahi_parameters(self, sahi_models_params: Dict[str, Dict]):
        """
        Set SAHI parameters for specific models.

        Args:
            sahi_models_params (Dict[str, Dict]): Dictionary of SAHI models and their parameters.
        """
        self.sahi_models_params = sahi_models_params

    def set_prompts(self, detection_class: List[str]):
        """
        Set new detection classes (prompts).

        Args:
            detection_class (List[str]): List of detection classes or prompts to be used for the detections.
        """
        self.detection_class = detection_class
        for model in self.models:
            if model.model_name != ModelNameRegistryDetection.TREX2.value:
                model.set_prompt(detection_class)

    def set_trex_input_class_bbox(self, trex_input_class_bbox: Dict[str, Union[str, List[int], str]]):
        """
        Set TREX input class bounding boxes.

        Args:
            trex_input_class_bbox (Dict[str, Union[str, List[int], str]]): TREX configuration for bounding boxes.
        """
        self.trex_input_class_bbox = trex_input_class_bbox

    def process_image(self) -> Dict[str, Dict[str, List]]:
        results = {}

        # Process each model
        # Ensure TREX2 runs last
        self.models.sort(key=lambda m: m.model_name == ModelNameRegistryDetection.TREX2.value)

        for model in self.models:
            model_name = model.model_name
            model.set_image(self.image)

            if model_name in self.sahi_models_params and SahiDetectionWrapper.is_supported(model_name):
                sahi_wrapper = SahiDetectionWrapper(model)
                sahi_params = self.sahi_models_params.get(model_name, {})
                sahi_wrapper.set_parameters(sahi_params)
                results[model_name] = sahi_wrapper.get_bbox(self.image)
            elif model_name == ModelNameRegistryDetection.TREX2.value:
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
            elif isinstance(config, str):
                model_result = results.get(config)
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

    def _save_image_with_boxes(self, result: Dict[str, List], output_path: str, model_name: str):
        """
        Save detection results to an image file, drawing bounding boxes and class labels.

        Args:
            result (Dict[str, List]): Detection result containing 'bboxes', 'labels', and 'scores'.
            output_path (str): File path to save the image.
            model_name (str): Name of the model for labeling purposes.
        """
        output_image = self.image.copy()
        colors = {}  # Store unique colors for each class
        for bbox, label, score in zip(result['bboxes'], result['labels'], result['scores']):
            # Assign a unique color to each class
            if label not in colors:
                np.random.seed(hash(label) % 2 ** 32)
                colors[label] = tuple(np.random.randint(0, 256, 3).tolist())
            color = colors[label]

            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{label} ({score:.2f}) [{model_name}]"
            cv2.putText(output_image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(output_path, output_image)
        print(f"Detection result saved to {output_path}")

    def save_results(self, results: Dict[str, Dict[str, List]], nms_results: Dict[str, List], output_dir: str):
        """
        Save detection results for each model and the combined NMS results.

        Args:
            results (Dict[str, Dict[str, List]]): Individual model results.
            nms_results (Dict[str, List]): Combined NMS results.
            output_dir (str): Directory to save the results.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save individual model results
        for model_name, result in results.items():
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            output_path = os.path.join(model_output_dir, f"{model_name}_result.jpg")
            self._save_image_with_boxes(result, output_path, model_name)

        # Save combined NMS results
        if 'bboxes' in nms_results and len(nms_results['bboxes']) > 0:
            nms_output_path = os.path.join(output_dir, "nms_result.jpg")
            self._save_image_with_boxes(nms_results, nms_output_path, "NMS")
        else:
            print("No NMS results to save.")


if __name__ == "__main__":
    # Image file
    image_path1 = "/home/nehoray/PycharmProjects/UniversaLabeler/data/tested_image/Soi/WhatsApp Image 2024-12-29 at 13.15.14 (4).jpeg"
    image_path2 = "/home/nehoray/PycharmProjects/UniversaLabeler/data/tested_image/Soi/WhatsApp Image 2024-12-29 at 13.15.14.jpeg"
    # Regular detection classes
    detection_classes = ["window", "holes", "doors", "balcony", "roof", "building", "tree"]

    # TREX input configuration (if needed)
    # trex_input_class_bbox = {
    #     "bus": ModelNameRegistryDetection.YOLO_WORLD.value,  # Source for "bus"
    #     "car": [471, 162, 514, 197],  # Manual bounding box for "car"
    # }
    # trex_input_class_bbox1 = {
    #     "window": [547,426,617,477],
    #     "window2":[400,96,417,149],
    #     "window3": [547,553,620,596]
    # }
    trex_input_class_bbox2 = {
        "window": MOST_CONFIDENCE,
        "holes": MOST_CONFIDENCE
    }
    # SAHI model parameters
    # sahi_model_params = {
    #     ModelNameRegistryDetection.YOLO_WORLD.value: {
    #         'slice_dimensions': (256, 256),
    #         'detection_conf_threshold': 0.7
    #     },
    #     ModelNameRegistryDetection.YOLO_ALFRED.value: {
    #         'slice_dimensions': (128, 128),
    #         'zoom_factor': 1.5
    #     }
    # }

    # Create an instance of ULDetection
    ul_detection = ULDetection(
        image_input=image_path2,
        detection_class=detection_classes,
        class_priorities={},
        model_priorities={ModelNameRegistryDetection.YOLO_WORLD.value: 2, ModelNameRegistryDetection.OPENGEOS.value: 1},
        use_nms=True,
        sahi_models_params=sahi_model_params,
        model_names=[ModelNameRegistryDetection.YOLO_WORLD.value,  #, ModelNameRegistryDetection.OPENGEOS.value,
                     ModelNameRegistryDetection.DINOX_DETECTION.value,
                     ModelNameRegistryDetection.TREX2.value,

                     ],
        filter_unwanted_classes=True,
        trex_input_class_bbox=trex_input_class_bbox2
    )

    # Set the image (if a different image needs to be loaded)
    # ul_detection.set_image(image_path2)

    # Update SAHI model parameters (if needed after initialization)
    # u can see the full sahi configuration in its file
    # ul_detection.set_sahi_parameters({
    #     ModelNameRegistryDetection.YOLO_WORLD.value: {
    #         'slice_dimensions': (512, 512),
    #         'detection_conf_threshold': 0.6
    #     }
    # })

    # # Update prompts (if detection classes need to be modified)
    # ul_detection.set_prompts(["car", "person", "bicycle"])
    #
    # # Update TREX inputs
    # ul_detection.set_trex_input_class_bbox({
    #     "car": [471, 162, 514, 197],  # Manual bounding box for "car"
    # })

    # Process the image
    nms_results, individual_results = ul_detection.process_image()

    # Save the results
    output_directory = "./test_soi4"
    ul_detection.save_results(individual_results, nms_results, output_directory)



