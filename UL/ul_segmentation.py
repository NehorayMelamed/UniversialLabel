import os
import cv2
import numpy as np
from typing import Dict, List, Union
from Factories.factory_segmentation_interface import FactorySegmentationInterface
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

        # Load models
        self.factory = FactorySegmentationInterface()
        self.models = []
        if model_names is None:
            model_names = self.factory.available_models()
        self.models = self._load_models(model_names)

        # Initialize SegSelector
        self.seg_selector = SegSelector()
        if self.class_priorities:
            self.seg_selector.set_class_priorities(self.class_priorities)
        if self.model_priorities:
            self.seg_selector.set_model_priorities(self.model_priorities)

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

    def load_models(self):
        """Load the necessary models from the FactorySegmentationInterface."""
        for model in self.models:
            model.init_model()

    def process_image(self) -> Dict[str, Dict[str, List]]:
        """
        Process the image with all models and format the results.

        Returns:
            formatted_result (Dict[str, List]): Final combined results after applying SegSelector.
            results (Dict[str, Dict[str, List]]): Individual model-specific results.
        """
        results = {}

        # Process each model
        for model in self.models:
            model_name = model.__class__.__name__
            model.set_image(self.image)

            # Ensure get_result is run before get_masks
            model.get_result()
            results[model_name] = model.get_masks()

        # Format class names to lowercase
        for model_name, result in results.items():
            result['labels'] = [label.lower() for label in result['labels']]

        # Apply SegSelector if required
        if self.use_segselector:
            image_shape = self.image.shape
            formatted_result_with_models = self.seg_selector.merge_results(results, image_shape)

            # Extract the final formatted result without model names
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
            # Use results directly if SegSelector is not used
            formatted_result_with_models = {
                model_name: {
                    'masks': result['masks'],
                    'labels': result['labels'],
                    'scores': [1.0] * len(result['masks'])  # Placeholder scores
                }
                for model_name, result in results.items()
            }
            formatted_result = {
                'masks': results['OpenEarthMapModel']['masks'],
                'labels': results['OpenEarthMapModel']['labels'],
                'scores': [1.0] * len(results['OpenEarthMapModel']['masks'])  # Placeholder scores
            }

        return formatted_result, formatted_result_with_models

    def save_results(self, formatted_result_with_models: Dict[str, Dict[str, List]], output_dir: str):
        """
        Save segmentation results for each model using the model's `save_colored_result` method.
        and also the final image with all the segmentations together.

        Args:
            formatted_result_with_models (Dict[str, Dict[str, List]]): Formatted segmentation results.
            output_dir (str): Directory to save the results.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save results for each model using its own save_colored_result
        for model_name, result in formatted_result_with_models.items():
            model_instance = self._get_model_instance_by_name(model_name)
            if model_instance is None:
                print(f"Model instance for {model_name} not found. Skipping save.")
                continue

            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # Call the model's save_colored_result to save its results
            output_path = os.path.join(model_output_dir, f"{model_name}_result.png")
            try:
                model_instance.save_colored_result(output_path)
                print(f"Saved results for {model_name} to {output_path}")
            except Exception as e:
                print(f"Error saving results for {model_name}: {e}")


    def _get_model_instance_by_name(self, model_name: str):
        """Get the model instance by its name."""
        for model in self.models:
            if model.__class__.__name__ == model_name:
                return model
        return None

if __name__ == "__main__":
    image_path = "/home/nehoray/PycharmProjects/test_opengeos/test_image.png"
    segmentation_class = ["road", "buildings", "pavement", "greenery"]
    ul_segmentation = ULSegmentation(
        image_input=image_path,
        segmentation_class=segmentation_class,
        class_priorities={},
        model_priorities={},
        use_segselector=True,
        model_names=[ModelNameRegistrySegmentation.SAM.value, ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
    )

    # Load the models
    ul_segmentation.load_models()

    # Process the image
    formatted_result, individual_results = ul_segmentation.process_image()

    # Save individual model results
    ul_segmentation.save_results(individual_results, "output_directory_models")

