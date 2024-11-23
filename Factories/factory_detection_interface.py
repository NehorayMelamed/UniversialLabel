from Factories.factory_model_interface import FactoryModelInterface
from ModelsFactory.Detection.REGEV_workspace.regev_model_detection import RegevDetectionModel
from ModelsFactory.Detection.facebook_detr_workspace.dert_detection_model import DetrDetectionModel
from ModelsFactory.Detection.google_vision_api_workspace.google_vision_api_detection_model import \
    GoogleVisionDetectionModel
from common.model_name_registry import ModelNameRegistryDetection, ConfigParameters
from ModelsFactory.Detection.GroundingDINO_workspace.grounding_dino_model import GroundingDINO_Model
from ModelsFactory.Detection.YOLO_WORLD_workspace.yolo_world_model import YOLOWorld_Model
from ModelsFactory.Detection.Alfred_detection_workspace.alfred_detection_nodel import AlfredDetectionModel
from ModelsFactory.Detection.Waldo_workspace.waldo_model_detection import WaldoDetectionModel
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from typing import Union


class FactoryDetectionInterface(FactoryModelInterface):
    """
    FactoryDetectionInterface creates detection models based on model names from ModelNameRegistry.
    """
    def __init__(self):
        # Map model names to their corresponding class implementations
        model_mapping = {
            ModelNameRegistryDetection.DINO.value: GroundingDINO_Model,
            ModelNameRegistryDetection.YOLO_WORLD.value: YOLOWorld_Model,
            ModelNameRegistryDetection.YOLO_ALFRED.value: AlfredDetectionModel,
            ModelNameRegistryDetection.WALDO.value: WaldoDetectionModel,
            ModelNameRegistryDetection.YOLO_REGEV.value: RegevDetectionModel,
            ModelNameRegistryDetection.DETR.value: DetrDetectionModel,
            ModelNameRegistryDetection.GOOGLE_VISION.value: GoogleVisionDetectionModel  # New entry for Google Vision

        }
        super().__init__(model_mapping)

    def create_model(self, model_type: Union[ModelNameRegistryDetection, str]) -> DetectionBaseModel:
        """
        Create a detection model based on the provided model type.

        Parameters:
        - model_type (Union[ModelNameRegistryDetection, str]): The type of model to create.

        Returns:
        - DetectionBaseModel: The instantiated model.

        Raises:
        - ValueError: If the model type is not valid.
        """
        if isinstance(model_type, str):
            model_type = ModelNameRegistryDetection(model_type)

        if model_type == ModelNameRegistryDetection.DINO:
            return GroundingDINO_Model(
                model_config_path=ConfigParameters.GROUNDING_DINO_config.value,
                model_checkpoint_path=ConfigParameters.GROUNDING_DINO_pth.value
            )
        elif model_type == ModelNameRegistryDetection.YOLO_WORLD:
            return YOLOWorld_Model(model_path=ConfigParameters.YOLO_WORLD_pt.value)
        elif model_type == ModelNameRegistryDetection.YOLO_ALFRED:
            return AlfredDetectionModel(model_path=ConfigParameters.YOLO_ALFRED_pt.value)
        elif model_type == ModelNameRegistryDetection.WALDO:
            return WaldoDetectionModel(model_path=ConfigParameters.YOLO_WALDO_pt.value)
        elif model_type == ModelNameRegistryDetection.YOLO_REGEV:
            return RegevDetectionModel(model_path=ConfigParameters.YOLO_REGEV_pt.value)
        elif model_type == ModelNameRegistryDetection.DETR:  # Add DETR handling
            return DetrDetectionModel(processor_path=ConfigParameters.DERT_MODEL.value,
                                    model_path=ConfigParameters.DERT_PROCESSOR.value)
        elif model_type == ModelNameRegistryDetection.GOOGLE_VISION:
            return GoogleVisionDetectionModel(credential_path=ConfigParameters.GOOGLE_VISION_KEY_API.value)

        else:
            raise ValueError(f"Unknown detection model type: {model_type}")

    def available_models(self) -> list:
        """
        Return a list of available detection model types.

        Returns:
        - list: A list of model names available in the ModelNameRegistry.
        """
        return [model.value for model in ModelNameRegistryDetection]

