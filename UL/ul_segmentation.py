import sys
sys.path.append("../")
import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_segmentation_interface import FactorySegmentationInterface
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from algorithms.seg_selector_backup import SegSelector
from common.model_name_registry import ModelNameRegistrySegmentation


class ULSegmentation:
    """
    A universal labeler segmentation class that processes an image using multiple segmentation models
    and combines the results using SegSelector if needed.
    """

    def __init__(self,
                 image_input: Union[str, np.ndarray],
                 segmentation_class: List[str],
                 class_priorities: Dict[str, int] = None,
                 model_priorities: Dict[str, int] = None,
                 use_segselector: bool = True,
                 model_names: List[Union[str, ModelNameRegistrySegmentation]] = None,
                 sam2_predict_on_bbox: List[np.ndarray] = None):
        self.image = self._load_image(image_input)
        self.segmentation_class = [cls.lower() for cls in segmentation_class]
        self.class_priorities = class_priorities if class_priorities else {}
        self.model_priorities = model_priorities if model_priorities else {}
        self.use_segselector = use_segselector
        self.sam2_predict_on_bbox = sam2_predict_on_bbox

        # Load models
        self.factory = FactorySegmentationInterface()
        if model_names is None:
            model_names = self.factory.available_models()

        # Ensure SAM2 is included if bounding boxes are provided
        if ModelNameRegistrySegmentation.SAM2.value not in model_names and sam2_predict_on_bbox is not None:
            model_names.append(ModelNameRegistrySegmentation.SAM2.value)

        self.models = self._load_models(model_names)

        # Initialize SegSelector
        self.seg_selector = SegSelector()
        if self.class_priorities:
            self.seg_selector.set_class_priorities(self.class_priorities)
        if self.model_priorities:
            self.seg_selector.set_model_priorities(self.model_priorities)

    def set_image(self, image_input: Union[str, np.ndarray]):
        """Update the image."""
        self.image = self._load_image(image_input)
        print("Image updated.")

    def set_classes(self, segmentation_class: List[str]):
        """Update segmentation classes."""
        self.segmentation_class = [cls.lower() for cls in segmentation_class]
        for model in self.models:
            model.set_prompt(self.segmentation_class)
        print("Segmentation classes updated.")

    def set_sam2_bboxes(self, bounding_boxes: List[np.ndarray]):
        """Update bounding boxes for SAM2 predictions."""
        self.sam2_predict_on_bbox = bounding_boxes
        print("SAM2 bounding boxes updated.")

    def add_model(self, model_name: Union[str, ModelNameRegistrySegmentation]):
        """Add a new model to the segmentation pipeline."""
        if model_name not in [model.model_name for model in self.models]:
            new_model = self.factory.create_model(model_name)
            new_model.init_model()
            new_model.set_prompt(self.segmentation_class)
            self.models.append(new_model)
            print(f"Model {model_name} added.")

    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """Load an image from a path or directly use the numpy array."""
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("Unsupported image input type. Must be a file path or a numpy array.")

    def _load_models(self, model_names: List[Union[str, ModelNameRegistrySegmentation]]) -> List:
        """Load the models from the factory."""
        models = []
        for model_name in model_names:
            model = self.factory.create_model(model_name)
            model.init_model()
            model.set_prompt(self.segmentation_class)
            models.append(model)
        return models

    def process_image(self) -> Dict[str, Dict[str, List]]:
        """
        Process the image with all models and format the results.
        """
        results = {}
        for model in self.models:
            model_name = model.model_name
            model.set_image(self.image)
            if model_name == ModelNameRegistrySegmentation.SAM2.value and self.sam2_predict_on_bbox:
                results[model_name] = model.get_result(boxes=self.sam2_predict_on_bbox)
            else:
                model.get_result()
                results[model_name] = model.get_masks()

        if self.use_segselector:
            image_shape = self.image.shape
            formatted_result_with_models = self.seg_selector.merge_results(results, image_shape)
            formatted_result = {
                'masks': [],
                'labels': [],
                'scores': []
            }
            for model_result in formatted_result_with_models.values():
                formatted_result['masks'].extend(model_result['masks'])
                formatted_result['labels'].extend(model_result['labels'])
                formatted_result['scores'].extend(model_result['scores'])
        else:
            formatted_result_with_models = {
                model_name: {
                    'masks': result['masks'],
                    'labels': result['labels'],
                    'scores': [1.0] * len(result['masks'])
                }
                for model_name, result in results.items()
            }
            formatted_result = {
                'masks': results['OpenEarthMapModel']['masks'],
                'labels': results['OpenEarthMapModel']['labels'],
                'scores': [1.0] * len(results['OpenEarthMapModel']['masks'])
            }

        return formatted_result, formatted_result_with_models

    def save_results(self, formatted_result_with_models: Dict[str, Dict[str, List]], output_dir: str):
        """
        Save segmentation results for each model using the model's `save_colored_result` method.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for model_name, result in formatted_result_with_models.items():
            model_instance = self._get_model_instance_by_name(model_name)
            if model_instance:
                output_path = os.path.join(output_dir, f"{model_name}_result.png")
                model_instance.save_colored_result(output_path)
                print(f"Saved results for {model_name} to {output_path}")

    def _get_model_instance_by_name(self, model_name: str):
        """Get the model instance by its name."""
        for model in self.models:
            if model.model_name == model_name:
                return model
        return None


if __name__ == "__main__":
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"
    segmentation_classes = ["car", "bus"]
    bounding_boxes = [
        np.array([468,157,518,203]),
        np.array([313,138,408,256]),
    ]

    ul_segmentation = ULSegmentation(
        image_input=image_path,
        segmentation_class=segmentation_classes,
        model_names=[ModelNameRegistrySegmentation.SAM2.value, ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value],
        sam2_predict_on_bbox=bounding_boxes
    )

    # Update image, classes, and bounding boxes
    ul_segmentation.set_image(image_path)
    ul_segmentation.set_classes(["person", "bicycle"])
    # ul_segmentation.set_sam2_bboxes([
    #     np.array([50, 60, 100, 150]),
    #     np.array([200, 220, 300, 350]),
    # ])

    # Process the image
    formatted_result, individual_results = ul_segmentation.process_image()

    # Save results
    ul_segmentation.save_results(individual_results, "test_output")


