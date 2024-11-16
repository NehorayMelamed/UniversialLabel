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

| Model Name     | Type        |
| :------------: | :---------: |
| DINO           | prompt      |
| YOLO_WORLD     | prompt      |
| YOLO_ALFRED    | classes     |
| WALDO          | prompt      |
| YOLO_REGEV     | classes     |
| OPEN_EARTH_MAP | classes     |
| SAM2           | none        |

### Explanation:

**DINO, YOLO_WORLD, WALDO**:
- These models use **prompts**, which means they expect descriptive text as input (e.g., "car," "person," "building") to define what objects to detect.

**YOLO_ALFRED, YOLO_REGEV, OPEN_EARTH_MAP**:
- These models use **classes**, meaning they are predefined with specific object categories and do not require a text prompt for detection. Instead, they look for the categories they were trained to recognize (e.g., vehicles, buildings, greenery).

**SAM2**:
- This model is labeled as **none**, implying that it does not utilize prompts or classes in the typical manner. It might instead be used for segmentation tasks without specific object prompts.

## Project Installation

### Prerequisites

- **Python**: Requires Python 3.8 or higher.
- **Platform**: The project was tested on Linux Ubuntu.

### Installing the Environment

1. Clone the repository:
    ```bash
    git clone <repo_url> UniversaLabeler
    ```

2. Update the `BASE_PROJECT_DIRECTORY_PATH` in `common/general_parameters.py`:
    ```python
    BASE_PROJECT_DIRECTORY_PATH = "/your/path/to/UniversaLabeler"
    ```

3. Install all required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Downloading Model Weights

- Place your **SECRET_KEY** in `UniversaLabeler/keys/secret_key.txt`.
- Run the script to download model weights:
    ```bash
    python UniversaLabeler/setup/download_pts.py
    ```
- Ensure the weights are downloaded successfully to:
  `UniversaLabeler/common/weights`

### Running Unit Tests

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

#### Detection

The `ULDetection` class allows combining detection models and using them with an optional NMS merging.

Example usage:

```python
from UL.ul_detection import ULDetection
from common.model_name_registry import ModelNameRegistryDetection

ul_detection = ULDetection(
    image_input="path/to/image.jpg",
    detection_classes=["SmallVehicle", "Person"],
    use_nms=True,
    model_names=[ModelNameRegistryDetection.YOLO_ALFRED.value, ModelNameRegistryDetection.WALDO.value]
)
formatted_result, individual_results = ul_detection.process_image()
```

#### Segmentation

The `ULSegmentation` class allows combining segmentation models and using them with an optional SegSelector for merging.

Example usage:

```python
from UL.ul_segmentation import ULSegmentation
from common.model_name_registry import ModelNameRegistrySegmentation

ul_segmentation = ULSegmentation(
    image_input="path/to/image.jpg",
    segmentation_class=["greenery", "buildings"],
    use_segselector=True,
    model_names=[ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value]
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

## NMS and SegSelector Explained

- **NMS Handler**: When using multiple detection models, the NMS handler combines overlapping bounding boxes from different models into a unified set, ensuring there are no duplicates.
- **SegSelector**: For segmentation tasks, the SegSelector class allows you to merge the results from multiple segmentation models based on a priority system. This is particularly useful if different segmentation models have different strengths.

## Safety Considerations

- All checkpoint files can be downloaded by specifying a **SECRET_KEY**.
- Ensure that your **SECRET_KEY** is saved to `UniversaLabeler/keys/secret_key.txt` to allow access to encrypted checkpoints.

## Planned Features

- **Model Enhancements**:
  - Support for more models including TREX and OPENGEOS.
  - Improvements to the SegSelector algorithm.
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
