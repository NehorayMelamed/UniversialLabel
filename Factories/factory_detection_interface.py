import os

import cv2

from Factories.factory_model_interface import FactoryModelInterface
from ModelsFactory.Detection.REGEV_workspace.regev_model_detection import RegevDetectionModel
from ModelsFactory.Detection.facebook_detr_workspace.dert_detection_model import DetrDetectionModel
from ModelsFactory.Detection.google_vision_api_workspace.google_vision_api_detection_model import \
    GoogleVisionDetectionModel
from ModelsFactory.Detection.GroundingDINO_workspace.grounding_dino_model import GroundingDINO_Model
from ModelsFactory.Detection.YOLO_WORLD_workspace.yolo_world_model import YOLOWorld_Model
from ModelsFactory.Detection.Alfred_detection_workspace.alfred_detection_nodel import AlfredDetectionModel
from ModelsFactory.Detection.Waldo_workspace.waldo_model_detection import WaldoDetectionModel
from ModelsFactory.Detection.DINO_X_workspace.dinox_detection_model import DINOXDetection  # Import new detection model
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from ModelsFactory.Detection.trex_workspace.trex_model_detection import TRex2DetectionModel
from common.model_name_registry import ModelNameRegistryDetection, ConfigParameters
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
            ModelNameRegistryDetection.GOOGLE_VISION.value: GoogleVisionDetectionModel,
            ModelNameRegistryDetection.DINOX_DETECTION.value: DINOXDetection  # New entry
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
                model_checkpoint_path=ConfigParameters.GROUNDING_DINO_pth.value,
            )
        elif model_type == ModelNameRegistryDetection.OPENGEOS:
            return GroundingDINO_Model(
                model_config_path=ConfigParameters.GROUNDING_DINO_OPENGEOS_config.value,
                model_checkpoint_path=ConfigParameters.GROUNDING_DINO_OPENGEOS_pth.value,
                model_name=ModelNameRegistryDetection.OPENGEOS.value
            )
        elif model_type == ModelNameRegistryDetection.YOLO_WORLD:
            return YOLOWorld_Model(model_path=ConfigParameters.YOLO_WORLD_pt.value)
        elif model_type == ModelNameRegistryDetection.YOLO_ALFRED:
            return AlfredDetectionModel(model_path=ConfigParameters.YOLO_ALFRED_pt.value)
        elif model_type == ModelNameRegistryDetection.WALDO:
            return WaldoDetectionModel(model_path=ConfigParameters.YOLO_WALDO_pt.value)
        elif model_type == ModelNameRegistryDetection.YOLO_REGEV:
            return RegevDetectionModel(model_path=ConfigParameters.YOLO_REGEV_pt.value)
        elif model_type == ModelNameRegistryDetection.DETR:
            return DetrDetectionModel(
                processor_path=ConfigParameters.DERT_MODEL.value,
                model_path=ConfigParameters.DERT_PROCESSOR.value
            )
        elif model_type == ModelNameRegistryDetection.GOOGLE_VISION:
            return GoogleVisionDetectionModel(credential_path=ConfigParameters.GOOGLE_VISION_KEY_API.value)
        elif model_type == ModelNameRegistryDetection.DINOX_DETECTION:
            # Read API token from the specified path in the registry
            api_key_path = ConfigParameters.DINOX_API_TOKEN.value
            if not os.path.exists(api_key_path):
                raise FileNotFoundError(f"DINO-X API key file not found at {api_key_path}")
            with open(api_key_path, "r") as file:
                api_token = file.read().strip()
            return DINOXDetection(api_token=api_token)
        elif model_type == ModelNameRegistryDetection.TREX2:
            # Read API token from the specified path in the registry
            api_key_path = ConfigParameters.TREX_API_TOKEN.value
            if not os.path.exists(api_key_path):
                raise FileNotFoundError(f"TREX-KEY API key file not found at {api_key_path}")
            with open(api_key_path, "r") as file:
                api_token = file.read().strip()
            return TRex2DetectionModel(token=api_token)
        else:
            raise ValueError(f"Unknown detection model type: {model_type}")

    def available_models(self) -> list:
        """
        Return a list of available detection model types.

        Returns:
        - list: A list of model names available in the ModelNameRegistry.
        """
        return [model.value for model in ModelNameRegistryDetection]



def main():
    # Initialize the factory
    detection_factory = FactoryDetectionInterface()

    ############### DINO-X ######################
    # # Specify the model type (DINOX_DETECTION in this case)
    # model_type = ModelNameRegistryDetection.DINOX_DETECTION.value
    #
    # # Create the DINOXDetection model
    # dinox_detection_model = detection_factory.create_model(model_type)
    #
    # # Initialize the model
    # dinox_detection_model.init_model()
    #
    # # Set prompts for the detection model
    # dinox_detection_model.set_prompt(["wheel", "eye", "helmet", "mouse", "mouth", "vehicle", "steering", "ear", "nose"])
    #
    # # Load and set the image
    # input_image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/DINO_X_workspace/DINO-X-API/assets/demo.png")
    # if input_image is None:
    #     raise ValueError("Failed to load input image. Check the file path.")
    #
    # dinox_detection_model.set_image(input_image)
    #
    # # Run inference and get results
    # print("Running inference...")
    # results = dinox_detection_model.get_result()
    #
    # # Retrieve formatted bounding boxes, labels, and scores
    # detection_boxes = dinox_detection_model.get_boxes()
    # print("Detection Results:")
    # print(detection_boxes)
    #
    # # Save the annotated image
    # output_path = "output/detected.png"  # Replace with the desired output path
    # dinox_detection_model.save_result(output_path)
    # print(f"Annotated image saved to {output_path}")

    # # Specify the model type (DINOX_DETECTION in this case)
    # model_type = ModelNameRegistryDetection.DINOX_DETECTION.value
    #
    # # Create the DINOXDetection model
    # dinox_detection_model = detection_factory.create_model(model_type)
    #
    # # Initialize the model
    # dinox_detection_model.init_model()
    #
    # # Set prompts for the detection model
    # dinox_detection_model.set_prompt(["wheel", "eye", "helmet", "mouse", "mouth", "vehicle", "steering", "ear", "nose"])
    #
    # # Load and set the image
    # input_image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/DINO_X_workspace/DINO-X-API/assets/demo.png")
    # if input_image is None:
    #     raise ValueError("Failed to load input image. Check the file path.")
    #
    # dinox_detection_model.set_image(input_image)
    #
    # # Run inference and get results
    # print("Running inference...")
    # results = dinox_detection_model.get_result()
    #
    # # Retrieve formatted bounding boxes, labels, and scores
    # detection_boxes = dinox_detection_model.get_boxes()
    # print("Detection Results:")
    # print(detection_boxes)
    #
    # # Save the annotated image
    # output_path = "output/detected.png"  # Replace with the desired output path
    # dinox_detection_model.save_result(output_path)
    # print(f"Annotated image saved to {output_path}")

    model_type = ModelNameRegistryDetection.TREX2.value
    trex2_detection_model = detection_factory.create_model(model_type)
    trex2_detection_model.init_model()
    # Define prompts with category names and bounding boxes
    # categories_with_boxes = {
    #     "bus": [[313,139,410,252]],  # Category: car
    #     "car": [[470,162,516,199]],  # Category: person
    # }
    categories_with_boxes = {
        "car": [[468.04302978515625, 159.61497497558594, 514.6063842773438, 199.15711975097656]],
    }

    # [468.04302978515625, 159.61497497558594, 514.6063842773438, 199.15711975097656]
    input_image = cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png")
    trex2_detection_model.set_image(input_image)
    trex2_detection_model.set_prompt(categories_with_boxes)
    trex2_detection_model.get_result()
    trex2_detection_model.save_result("output_directory/annotated_image.jpg")






    # model = TRex2DetectionModel(token="9d01f376fbf2bb13123c85a82da8e154")
    # model.init_model()
    #
    # # Set the image
    # input_image = cv2.imread("/home/nehoray/PycharmProjects/tests_wotkspace/T-Rex/assets/trex2_api_examples/interactive1.jpeg")
    # model.set_image(input_image)
    #
    # # Define prompts with category names and bounding boxes
    # categories_with_boxes = {
    #     "car": [[100, 100, 200, 200], [300, 300, 400, 400]],  # Category: car
    #     "person": [[150, 150, 250, 250]],  # Category: person
    # }
    # model.set_prompt(categories_with_boxes)
    #
    # # Run inference
    # results = model.get_result()
    # print("Results:", results)
    #
    # # Save annotated image
    # output_path = "output_directory/annotated_image.jpg"
    # model.save_result(output_path)












if __name__ == "__main__":
    main()
