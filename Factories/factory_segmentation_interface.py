import sys
import cv2
import numpy as np

from common.model_name_registry import ModelNameRegistrySegmentation, ConfigParameters
sys.path.append(ConfigParameters.SAM2_PATH_CONFLICT_TO_APPEND.value)

from Factories.factory_model_interface import FactoryModelInterface
from ModelsFactory.Segmentation.SAM2_workspace.sam_2_model import SAM2Segmentation
from ModelsFactory.Segmentation.SAM_workspace.sam_model import SegmentationBaseModel, SAMSegmentation
from ModelsFactory.Segmentation.open_earth_map_workspace.open_earth_map_model import OpenEarthMapModel
from ModelsFactory.Segmentation.DINO_x_workspace.dino_x_segmentation_model import DINOXSegmentation
import os


class FactorySegmentationInterface(FactoryModelInterface):
    """
    FactorySegmentationInterface implements the FactoryModelInterface.
    It returns segmentation models such as GroundingDINO, SAM2, OpenEarthMapModel, and DINOXSegmentation.
    """

    def __init__(self):
        model_mapping = {
            ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value: OpenEarthMapModel,
            ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value: DINOXSegmentation,
            ModelNameRegistrySegmentation.SAM2.value: SAM2Segmentation,
        }
        super().__init__(model_mapping)

    def create_model(self, model_type: str) -> SegmentationBaseModel:
        """
        Create and return a segmentation model based on the model_type string.

        Parameters:
        - model_type (str): The type of model to create (e.g., "OpenEarthMap", "DINOX_SEGMENTATION")

        Returns:
        - SegmentationBaseModel: The initialized segmentation model.
        """
        if model_type == ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value:
            model = OpenEarthMapModel(
                model_dir=ConfigParameters.OPEN_EARTH_MAP_MODEL_DIR.value,
                model_checkpoint_name=ConfigParameters.OPEN_EARTH_MAP_MODEL_NAME.value
            )
            model.init_model()
            return model
        elif model_type == ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value:
            # Read API token for DINO-X Segmentation
            api_key_path = ConfigParameters.DINOX_API_TOKEN.value
            if not os.path.exists(api_key_path):
                raise FileNotFoundError(f"DINO-X API key file not found at {api_key_path}")
            with open(api_key_path, "r") as file:
                api_token = file.read().strip()
            model = DINOXSegmentation(api_token=api_token)
            model.init_model()
            return model
        elif model_type == ModelNameRegistrySegmentation.SAM.value:
            model = SAMSegmentation(checkpoint_path=ConfigParameters.SAM_THIN_PT.value)
            model.init_model()
            return model
        elif model_type == ModelNameRegistrySegmentation.SAM2.value:
            model = SAM2Segmentation(checkpoint_path=ConfigParameters.SAM2_THIN_PT.value,
                                     model_cfg=ConfigParameters.SAM2_THIN_CONFIG.value)
            model.init_model()
            return model

        else:
                raise ValueError(f"Unknown segmentation model type: {model_type}")

    def available_models(self) -> list:
        """
        Return a list of available segmentation model types.

        Returns:
        - list: A list of model names available in the ModelNameRegistry.
        """
        return [model.value for model in ModelNameRegistrySegmentation]


if __name__ == '__main__':
    image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png")

    fc = FactorySegmentationInterface()

    # Define multiple bounding boxes
    bounding_boxes = [
        np.array([45, 82, 123, 181]),
        np.array([470, 160, 513, 199]),
        np.array([305, 136, 427, 255]),
    ]
    sam2 = fc.create_model(ModelNameRegistrySegmentation.SAM2.value)
    sam2.set_image(image)
    # Perform segmentation for multiple bounding boxes
    results = sam2.get_result(boxes=bounding_boxes)

    # Save results
    sam2.save_colored_result(results, "output/sam2_results.jpg")

    a = sam2.get_bbox_from_masks()
