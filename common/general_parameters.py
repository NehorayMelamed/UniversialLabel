import os
from pathlib import Path
BASE_PROJECT_DIRECTORY_PATH = Path(__file__).parent.parent
__KEYS_DIRECTORY_PATH = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "keys")
SECRET_KEY_PATH = os.path.join(__KEYS_DIRECTORY_PATH, "secret_key.txt")
ENCRYPTED_UTL_PATH = os.path.join(__KEYS_DIRECTORY_PATH, "encrypted_url.txt")

__COMMON_PATH = os.path.join(BASE_PROJECT_DIRECTORY_PATH, "common")
WEIGHTS_PATH = os.path.join(__COMMON_PATH, "weights")
DOWNLOAD_PTS_FILE_PATH = os.path.join(WEIGHTS_PATH, "pts_download.zip")



## for unit test data
TEST_IMAGE_SKY_DETECTION_PATH = os.path.join(BASE_PROJECT_DIRECTORY_PATH,"data", "tested_image", "detection", "from_sky.jpeg")
TEST_IMAGE_STREET_DETECTION_PATH = os.path.join(BASE_PROJECT_DIRECTORY_PATH,"data", "tested_image", "detection", "street.png")