import os
from enum import Enum

from common.general_parameters import WEIGHTS_PATH, BASE_PROJECT_DIRECTORY_PATH


class ModelNameRegistryDetection(Enum):
    # Detection Models
    DINO = "DINO"
    YOLO_WORLD = "YOLO_WORLD"
    YOLO_ALFRED = "YOLO_ALFRED"
    WALDO = "WALDO"
    YOLO_REGEV = "YOLO_REGEV"
    DETR = "DETR"  # New entry
    GOOGLE_VISION = "GOOGLE_VISION"  # New entry for Google Vision model

class ModelNameRegistrySegmentation(Enum):
    # Segmentation Models
    OPEN_EARTH_MAP = "OPEN_EARTH_MAP"
    SAM2 = "SAM2"




class ConfigParameters(Enum):
    GROUNDING_DINO_config = os.path.join(BASE_PROJECT_DIRECTORY_PATH,"ModelsFactory","Detection","GroundingDINO_workspace","git_workspace","GroundingDINO","groundingdino","config","GroundingDINO_SwinT_OGC.py")
    print(GROUNDING_DINO_config)
    GROUNDING_DINO_pth = os.path.join(BASE_PROJECT_DIRECTORY_PATH,"ModelsFactory","Detection","GroundingDINO_workspace","git_workspace","GroundingDINO","weights","groundingdino_swint_ogc.pth")
    print(GROUNDING_DINO_pth)
    YOLO_WALDO_pt = os.path.join(WEIGHTS_PATH, "WALDO30_yolov8m_640x640.pt")
    YOLO_ALFRED_pt = os.path.join(WEIGHTS_PATH, "alfred_best.pt")
    YOLO_WORLD_pt = os.path.join(WEIGHTS_PATH,"yolov8s-world.pt")
    SAM2_CHECKPOINT_PATH = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "ModelsFactory","Segmentation","SAM2_workspace","git_workspace","sam2","sam2","configs","sam2","sam2_hiera_l.yaml")
    SAM2_CONFIG_FILE = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "ModelsFactory","Segmentation","SAM2_workspace","git_workspace","sam2","sam2","configs","sam2","sam2_hiera_t.yaml")
    YOLO_REGEV_pt = os.path.join(WEIGHTS_PATH, "regev_model.pt")
    DERT_MODEL =    os.path.join(WEIGHTS_PATH, "facebook_detr", "models--facebook--detr-resnet-50","snapshots","70120ba84d68ca1211e007c4fb61d0cd5424be54"  )
    DERT_PROCESSOR =os.path.join(WEIGHTS_PATH, "facebook_detr", "models--facebook--detr-resnet-50","snapshots","70120ba84d68ca1211e007c4fb61d0cd5424be54" )
    OPEN_EARTH_MAP_MODEL_DIR = os.path.join(WEIGHTS_PATH,)
    OPEN_EARTH_MAP_MODEL_NAME = "open_earth_map_model.pth"
    GOOGLE_VISION_KEY_API = os.path.join(os.path.join(BASE_PROJECT_DIRECTORY_PATH, "keys", "google_vision_api_key", os.listdir(os.path.join(BASE_PROJECT_DIRECTORY_PATH, "keys", "google_vision_api_key"))[0]) )

    @classmethod
    def has_value(cls, value):
        """
        Check if the given value is a valid model name in the registry.

        Parameters:
        - value (str): The string value to check.

        Returns:
        - bool: True if the value is a valid model name, False otherwise.
        """
        return value in cls._value2member_map_

PROMPT_MODEL = "free prompt"