# UniversaLabeler

## Overview

UniversaLabeler is a project that provides a modular and extensible platform for implementing detection and segmentation models. The goal of the project is to simplify the process of labeling images, by making use of sophisticated models for both detection and segmentation tasks, and to provide an easy-to-use interface to customize and extend model capabilities.

### Key Features

- **Universal Labeling Interface**: Combines multiple models for detection and segmentation, allowing you to work with a unified interface.
- **SegSelector**: Allows merging results from multiple segmentation models based on priority configurations.
- **NMS Handler**: Custom Non-Maximum Suppression for bounding box merging.
- **Factory Pattern for Model Management**: Easily manage detection and segmentation models using the factory design pattern.


## Available Models and Their Types

Below is the list of available models in the UniversaLabeler project, along with their type:

|  Model Name   | Class/prompt |  model type  |
|:-------------:|:------------:|:------------:|
|   GrouDino    |    prompt    |  Detection   |
|  Yolo world   |    prompt    |  Detection   |
|  Yolo Alfred  |   classes    |  Detection   |
|     Waldo     |    prompt    |  Detection   |
|  Yolo regev   |   classes    |  Detection   |
| OpenEarthMap  |   classes    | Segmentation |
|     SAM2      |  bbox/none   | Segmentation |
| Google vision |    prompt    |  Detection   |
|      SAM      |     none     | Segmentation |   
|    Dino-X     |    prompt    |  Detection   |   
|    Dino-X     |    prompt    | Segmentation |   
|   OpenGeos    |    prompt    |  Detection   |
|     TREX      |  reference   |  Detection   |   


### Explanation:

- The models use **prompts**, which means they expect descriptive text as input (e.g., "car," "person," "building") to define what objects to detect.

- The models use **classes**, meaning they are predefined with specific object categories and do not require a text prompt for detection. Instead, they look for the categories they were trained to recognize (e.g., vehicles, buildings, greenery).

**SAM2**:
- This model is labeled as **none**, implying that it does not utilize prompts or classes in the typical manner. It might instead be used for segmentation tasks without specific object prompts.

## Project Installation

### Prerequisites

- **Python**: Requires Python 3.8 or higher.
- **Platform**: The project was tested on Linux Ubuntu.

### Installing the Environment

!! my advice - go for Conda (its fix some issues which may appear while using venv and pip) !!!

1.  #### Clone the repository:
    ```bash
    git clone https://github.com/NehorayMelamed/UniversialLabel.git
    ```


2. #### Install all required packages:
    1. Install Torch base on your OS (https://pytorch.org/get-started/locally/)
    2. install other packages
   ```bash
    pip install -r requirements.txt
    ```
    
3. #### Install GroundingDINO 
    !!! no needed anymore(see in  requirements.txt) !!!

    (base on their installation guid - https://github.com/IDEA-Research/GroundingDIN )
    1. cd ModelsFactory/Detection/GroundingDINO_workspace/git_workspace/GroundingDINO/
    2. pip install -e .


4. #### Install Mega cli 
    base on their website

    https://mega.io/cmd#download


6. #### Install SAM
    !!! no needed anymore(see in  requirements.txt) !!!

    note that SAM has some issues during its installation, if ur facing with something like "ModuleNotFoundError: No module named '_bz2'"
    please visit the next page -> https://stackoverflow.com/questions/12806122/missing-python-bz2-module
    *see also the /issues/ModuleNotFoundError/bz2*

### Download Model Weights

- Place your **SECRET_KEY** in `UniversaLabeler/keys/secret_key.txt`.
- Run the script to download model weights and extract them into the right place:
    ```bash
    python UniversaLabeler/setup/download_pts.py
    ```
- Ensure the weights are downloaded successfully to:
  `UniversaLabeler/common/weights`

### Run Unit Tests

- Navigate to `UniversaLabeler/TestWorkspace` and run the unit tests using:
  ```bash
  python tests_main.py
  ```
- You can also run individual tests from:
  ```
  UniversaLabeler/TestWorkspace/test_universal_labeler
  UniversaLabeler/TestWorkspace/test_factory_and_models
  ```

## Usage

### Universal Labeler Detection and Segmentation

### Detection

The `ULDetection` class allows combining detection models and using them with an optional NMS merging.

Example usage:

```python
from UL.ul_detection import ULDetection
from common.model_name_registry import ModelNameRegistryDetection
    
detection_classes = ["tree", "grass", "car", "person"]

ul_detection = ULDetection(
    image_input=image_path,
    detection_class=detection_classes,
    class_priorities={},
    model_priorities={},
    sahi_models_params=sahi_model_params,
    sahi_models_params={},
    model_names=[ModelNameRegistryDetection.YOLO_WORLD, ModelNameRegistryDetection.YOLO_ALFRED]
)
formatted_result, individual_results = ul_detection.process_image()
```

### Segmentation

The `ULSegmentation` class allows combining segmentation models and using them with an optional SegSelector for merging.

Example usage:

```python
from UL.ul_segmentation import ULSegmentation
from common.model_name_registry import ModelNameRegistrySegmentation

    segmentation_class = ["road", "buildings", "pavement", "greenery"]
    ul_segmentation = ULSegmentation(
        image_input=image_path,
        segmentation_class=segmentation_class,
        class_priorities={},
        model_priorities={},
        use_segselector=True,
        model_names=[ModelNameRegistrySegmentation.SAM.value, ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
    )
formatted_result, individual_results = ul_segmentation.process_image()
```

### Available Classes and Models

#### Detection and Segmentation Classes

You can use predefined classes for detection and segmentation provided in `common/classes.py`. Example classes:

```python
class DetectionClassesName(Enum):
    SmallVehicle = "SmallVehicle"  # Alfred
    BigVehicle = "BigVehicle"      # Alfred
    Person = "Person"              # Waldo

class SegmentationClassesName(Enum):
    greenery = "greenery"          # Open Earth Map
    buildings = "buildings"        # Open Earth Map
    road = "road"                  # Open Earth Map
```

#### Available Models

Available detection and segmentation models are listed in `common/model_name_registry.py`.

```python
class ModelNameRegistryDetection(Enum):
    DINO = "DINO"
    YOLO_WORLD = "YOLO_WORLD"
    YOLO_ALFRED = "YOLO_ALFRED"

class ModelNameRegistrySegmentation(Enum):
    OPEN_EARTH_MAP = "OPEN_EARTH_MAP"
    SAM2 = "SAM2"
```

### NMS and SegSelector Explained

- **NMS Handler**: When using multiple detection models, the NMS handler combines overlapping bounding boxes from different models into a unified set, ensuring there are no duplicates.
- **SegSelector**: For segmentation tasks, the SegSelector class allows you to merge the results from multiple segmentation models based on a priority system. This is particularly useful if different segmentation models have different strengths.

### Safety Considerations

- All checkpoint files can be downloaded by specifying a **SECRET_KEY**.
- For now its no needed anymore:)
- Ensure that your **SECRET_KEY** is saved to `UniversaLabeler/keys/secret_key.txt` to allow access to encrypted checkpoints.
    
## Notes :
### Google api usage
if you are going to use Google api, please:
1. Generate the key for the "Service Account Key" of google  for *vision api* from your project, or use our Key.  (https://cloud.google.com/iam/docs/keys-create-delete)
2. create a directory inside the keys directory with the name - google_vision_api_key 
```                                         
cd keys   
```     
```                                         
mkdir google_vision_api_key 
```      
3. and place there the json key from google


## Issues
1. for the SEEM, i got some issue with the -  pip install mpi4py
in the end i install it using the conda conda install mpi4py
2. for the "ModuleNotFoundError: No module named '_bz2'"
    see above in the SAM issues section.


## New and Last Updates
- nehoray - support pass sahi with costume parameters per models - see UL class
- OpenGeos model was added
- SAM was added
- Dino-X was added 
- save multy segmentations models results


## Planned Features

- **Model Enhancements**:
  - Support for more models including TREX and SAM2.
  - Make the Sahi works with other models
  - Improvements to the SegSelector algorithm.
  - Support classes_cleaner - which will return only the desired user input classes
  - Support auto Models selector base on the angle 
  
- **Docker Support**: Background Docker integration for running models as services.
- **Testing**: Additional unit tests for model reliability.

## Summary

- **Clone the Repository**.
- **Configure BASE_PROJECT_DIRECTORY_PATH**.
- **Install Requirements**.
- **Download Model Weights** with your **SECRET_KEY**.
- **Run Unit Tests** and verify that the setup works properly.
- **Use Universal Labeler** for detection and segmentation tasks as shown in the usage examples.

Feel free to add more diagrams and visual aids to understand the project flow better, such as a flowchart showing how data is processed from the initial image input to final labeling results.
"""
### Feel free to make a contant with me - Nehoray Melamed - +972 053 532 7656
