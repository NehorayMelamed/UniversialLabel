from Factories.factory_model_interface import FactoryModelInterface
from ModelsFactory.Segmentation.SAM_workspace.sam_model import SegmentationBaseModel, SAMSegmentation
from ModelsFactory.Segmentation.open_earth_map_workspace.open_earth_map_model import OpenEarthMapModel
from ModelsFactory.Segmentation.DINO_x_workspace.dino_x_segmentation_model import DINOXSegmentation
from common.model_name_registry import ModelNameRegistrySegmentation, ConfigParameters
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
            # ModelNameRegistrySegmentation.SAM2.value: SAM2Model,
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
        else:
            raise ValueError(f"Unknown segmentation model type: {model_type}")

    def available_models(self) -> list:
        """
        Return a list of available segmentation model types.

        Returns:
        - list: A list of model names available in the ModelNameRegistry.
        """
        return [model.value for model in ModelNameRegistrySegmentation]


