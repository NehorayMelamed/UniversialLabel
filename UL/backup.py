import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_segmentation_interface import FactorySegmentationInterface
from common.model_name_registry import ModelNameRegistrySegmentation
from algorithms.seg_selector import SegSelector


class UniversalLabelerSegmentation:
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
                 model_names: List[Union[str, ModelNameRegistrySegmentation]] = None):
        """
        Initialize the UniversalLabelerSegmentation class.

        Args:
            image_input (Union[str, np.ndarray]): Input image either as a path or a numpy array.
            segmentation_class (List[str]): List of segmentation classes or prompts to be used.
            class_priorities (Dict[str, int]): Dictionary of class priorities for SegSelector.
            model_priorities (Dict[str, int]): Dictionary of model priorities for SegSelector.
            use_segselector (bool): Flag indicating whether to use SegSelector.
            model_names (List[Union[str, ModelNameRegistrySegmentation]]): List of model names. Defaults to None.
        """
        self.image = self._load_image(image_input)
        self.segmentation_class = [cls.lower() for cls in segmentation_class]
        self.class_priorities = class_priorities if class_priorities else {}
        self.model_priorities = model_priorities if model_priorities else {}
        self.use_segselector = use_segselector

        self.factory = FactorySegmentationInterface()
        self.models = []
        if model_names is None:
            model_names = self.factory.available_models()
        self.models = self._load_models(model_names)

        self.seg_selector = SegSelector()
        if self.class_priorities:
            self.seg_selector.set_class_priorities(self.class_priorities)
        if self.model_priorities:
            self.seg_selector.set_model_priorities(self.model_priorities)

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

    def _load_models(self, model_names: List[Union[str, ModelNameRegistrySegmentation]]) -> List:
        """
        Load the models from the factory.
        """
        models = []
        for model_name in model_names:
            model = self.factory.create_model(model_name)
            model.init_model()
            model.set_prompt(self.segmentation_class)
            models.append(model)
        return models

    def load_models(self):
        """
        Load the necessary models from the FactorySegmentationInterface.
        """
        for model in self.models:
            model.init_model()

    def process_image(self) -> Dict[str, Dict[str, List]]:
        """
        Process the image with all models and format the results.
        """
        results = {}

        # Process each model
        for model in self.models:
            model_name = model.__class__.__name__
            model.set_image(self.image)
            results[model_name] = model.get_masks()

        # Format class names to lowercase
        for model_name, result in results.items():
            result['labels'] = [label.lower() for label in result['labels']]

        # Apply SegSelector if required
        if self.use_segselector:
            formatted_result = self.seg_selector.merge_results(results)
        else:
            # Use results directly if SegSelector is not used
            formatted_result = {
                "OpenEarthMapModel": {
                    'masks': results['OpenEarthMapModel']['masks'],
                    'labels': results['OpenEarthMapModel']['labels'],
                    'scores': [1.0] * len(results['OpenEarthMapModel']['masks'])  # Placeholder scores
                }
            }

        return formatted_result, results

    def save_results(self, formatted_result: Dict[str, Dict[str, List]], output_dir: str):
        """
        Save segmentation results for each model and the final SegSelector result.

        Args:
            formatted_result (Dict[str, Dict[str, List]]): Formatted segmentation results.
            output_dir (str): Directory to save the results.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save each model's result
        for model_name, result in formatted_result.items():
            model_output_dir = os.path.join(output_dir, model_name)
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)

            output_path = os.path.join(model_output_dir, f"{model_name}_result.png")

            if model_name == "OpenEarthMapModel":
                # Use the OpenEarthMapModel's save_colored_result method
                model_instance = self._get_model_instance_by_name(model_name)
                if model_instance:
                    model_instance.save_colored_result(output_path)

    def _get_model_instance_by_name(self, model_name: str):
        """
        Get the model instance by its name.

        Args:
            model_name (str): The name of the model.

        Returns:
            The model instance if found, otherwise None.
        """
        for model in self.models:
            if model.__class__.__name__ == model_name:
                return model
        return None


# Example usage
if __name__ == "__main__":
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/images/mix/small_car.jpeg"
    segmentation_class = ["road", "buildings", "pavement", "greenery"]
    ul_segmentation = UniversalLabelerSegmentation(
        image_input=image_path,
        segmentation_class=segmentation_class,
        class_priorities={"road": 2, "buildings": 1},
        model_priorities={ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value: 2},
        use_segselector=False,  # Skip using SegSelector for now
        model_names=[ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
    )

    # Load the models
    ul_segmentation.load_models()

    # Process the image
    formatted_result, _ = ul_segmentation.process_image()

    # Save the results
    ul_segmentation.save_results(formatted_result, "output_directory")
