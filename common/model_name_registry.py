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
    DINOX_DETECTION = "DINOX_DETECTION"
    OPENGEOS = "OPENGEOS"
    TREX2 = "TREX2"
class ModelNameRegistrySegmentation(Enum):
    # Segmentation Models
    DINOX_SEGMENTATION = "DINOX_SEGMENTATION"
    SAM = "SAM"
    OPEN_EARTH_MAP = "OPEN_EARTH_MAP"
    SAM2 = "SAM2"
    FASTSAM = "FASTSAM"
    SEEM = "SEEM"




class ConfigParameters(Enum):
    GROUNDING_DINO_config = os.path.join(BASE_PROJECT_DIRECTORY_PATH,"ModelsFactory","Detection","GroundingDINO_workspace","git_workspace","GroundingDINO","groundingdino","config","GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_pth = os.path.join(WEIGHTS_PATH,"groundingdino_swint_ogc.pth")
    GROUNDING_DINO_OPENGEOS_config = os.path.join(BASE_PROJECT_DIRECTORY_PATH,"ModelsFactory","Detection","GroundingDINO_workspace","git_workspace","GroundingDINO","groundingdino","config","GroundingDINO_SwinB_cfg.py")
    GROUNDING_DINO_OPENGEOS_pth = os.path.join(WEIGHTS_PATH,"groundingdino_swinb_cogcoor.pth")

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
    DINOX_API_TOKEN = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "keys", "dinox_key")
    TREX_API_TOKEN = DINOX_API_TOKEN
    SAM_THIN_PT = os.path.join(WEIGHTS_PATH, "sam_vit_b_01ec64.pth")
    SAM_LARGE_PT = os.path.join(WEIGHTS_PATH, "sam_vit_b_01ec64.pth")

    SAM2_THIN_PT = os.path.join(WEIGHTS_PATH,"sam2.1_hiera_large.pt")
    SAM2_THIN_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    SAM2_PATH_CONFLICT_TO_APPEND = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "ModelsFactory","Segmentation","SAM2_workspace", "git_workspace", "sam2")

    SEEM_WEIGHTS_FILE_PATH = os.path.join(WEIGHTS_PATH, "seem_focall_v1.pt")
    SEEM_CONFIG_FILE_PATH = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "ModelsFactory", "Segmentation", "SEEM_workspace", "git_workspace", "configs", "seem", "focall_unicl_lang_v1.yaml")

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
PROMPT_MODEL_BASE_INFERENCE_BBOX = "PROMPT_MODEL_BASE_INFERENCE_BBOX"
MOST_CONFIDENCE = "MOST_CONFIDENCE"