from Factories.factory_model_interface import FactoryModelInterface
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from ModelsFactory.Segmentation.open_earth_map_workspace.open_earth_map_model import OpenEarthMapModel
from common.model_name_registry import ModelNameRegistrySegmentation, ConfigParameters


class FactorySegmentationInterface(FactoryModelInterface):
    """
    FactorySegmentationInterface implements the FactoryModelInterface.
    It returns segmentation models such as GroundingDINO, SAM2, and OpenEarthMapModel.
    """

    def create_model(self, model_type: str) -> SegmentationBaseModel:
        """
        Create and return a segmentation model based on the model_type string.

        Parameters:
        - model_type (str): The type of model to create (e.g., "GroundingDINO", "SAM2", "OpenEarthMap")

        Returns:
        - SegmentationBaseModel: The initialized segmentation model.
        """
        # if model_type == ModelNameRegistrySegmentation.SAM2:
        #     raise NotImplemented
        if model_type == ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value:
            model = OpenEarthMapModel(model_dir=ConfigParameters.OPEN_EARTH_MAP_MODEL_DIR.value, model_checkpoint_name=ConfigParameters.OPEN_EARTH_MAP_MODEL_NAME.value)
            model.init_model()
            return model
        else:
            raise ValueError(f"Unknown segmentation model type: {model_type}")

    def available_models(self) -> list:
        """
        Return a list of available detection model types.

        Returns:
        - list: A list of model names available in the ModelNameRegistry.
        """
        return [model.value for model in ModelNameRegistrySegmentation]


