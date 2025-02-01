# Extending UniversaLabeler

## Introduction

Extending **UniversaLabeler** allows developers to integrate new models seamlessly into the system, making them fully functional alongside existing models. This guide will walk you through the necessary steps to add a new model, ensuring compatibility with the framework.

## Adding a New Model

To add a new model, follow these structured steps. For this guide, we will assume the model being added is named `DLModel` and that it is a **detection** model.

### Directory Structure

All models should be added under their respective **ModelFactory** directory:

```
UniversalLabeler/
├── ModelsFactory/
│   ├── Detection/
│   │   ├── DL_model_workspace/
│   │   │   ├── Dl_detection_model.py
│   │   │   ├── git_workspace/  # (Only if Git cloning is required)
│   ├── Segmentation/
│   ├── Captioning/
```

### Step 1: Implementing the Model Class

Under `ModelsFactory/Detection/DL_model_workspace/`, create a new file named `Dl_detection_model.py` and define the model class.

The new class **must** inherit from `BaseDetectionModel` to ensure compatibility with the system.

#### Example Implementation

```python
import os
from abc import abstractmethod
from typing import Dict, List
import cv2
from ModelsFactory.base_model import BaseDetectionModel

class DLDetectionModel(BaseDetectionModel):
    """
    DLDetectionModel implements a new detection model following UniversaLabeler’s standards.
    """
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model_name = ModelNameRegistryDetection.DL.value  # Assign model name
        self.init_model()

    def init_model(self):
        """Initialize the model and load weights"""
        self.model = self.load_model(self.model_path)

    @abstractmethod
    def set_prompt(self, prompt: List[str]):
        pass

    @abstractmethod
    def set_image(self, image):
        pass

    @abstractmethod
    def get_result(self):
        pass

    def get_boxes(self) -> Dict[str, List]:
        """Returns detected bounding boxes, labels, and scores"""
        return {
            "bboxes": [],
            "labels": [],
            "scores": []
        }
```

### Step 2: Registering the Model in the Data Classes

Modify the `common/model_name_registry.py` file to include the new model in `ModelNameRegistryDetection`.

```python
from enum import Enum

class ModelNameRegistryDetection(Enum):
    # Existing models
    DINO = "DINO"
    YOLO_WORLD = "YOLO_WORLD"
    
    # Add the new model
    DL = "DL"
```

### Step 3: Configuring the Factory Interface

To integrate the model into the **Factory Interface**, update `Factories/factory_detection_interface.py`.

#### 3.1: Add Model Mapping

Inside the `model_mapping` dictionary, add:

```python
ModelNameRegistryDetection.DL.value: DLDetectionModel
```

#### 3.2: Modify `create_model` Method

Update the `create_model` function to recognize the new model:

```python
elif model_type == ModelNameRegistryDetection.DL:
    return DLDetectionModel(model_path=ConfigParameters.DL.value)
```

### Step 4: Define Configuration Parameters

If your model requires additional configuration such as weights paths, update `common/model_name_registry.py`:

```python
class ConfigParameters:
    DL = "path/to/dl_model_weights.pt"
```

### Step 5: Testing the Model

Before using the new model, it is recommended to test its integration. Run the following to verify the model loads correctly:

```python
from Factories.factory_detection_interface import FactoryDetectionInterface
from common.model_name_registry import ModelNameRegistryDetection

factory = FactoryDetectionInterface()
model = factory.create_model(ModelNameRegistryDetection.DL.value)
assert model is not None, "Model creation failed!"
print("DLDetectionModel successfully integrated.")
```

### Step 6: Using the New Model in ULDetection

Once the model is integrated, you can now use it within `ULDetection` by adding it to `model_names`:

```python
from universal_labeler.ul_detection import ULDetection
from common.model_name_registry import ModelNameRegistryDetection

ul_detection = ULDetection(
    image_input="path/to/image.jpg",
    detection_class=["car", "bus"],
    model_names=[ModelNameRegistryDetection.DL.value, ModelNameRegistryDetection.OPENGEOS.value]
)
nms_results, individual_results = ul_detection.process_image()
```

### Step 7: Advanced Model-Specific Parameters

For models requiring unique parameters like `trex_input_class_bbox`, additional logic can be implemented within the UL classes. Refer to **ULDetection** or **ULSegmentation** documentation for guidance.

---

## Summary

Adding a new model to UniversaLabeler involves:
1. **Creating the model class** and inheriting from `BaseDetectionModel`.
2. **Registering the model name** in `ModelNameRegistryDetection`.
3. **Integrating the model into the Factory Interface**.
4. **Defining configuration parameters** (if needed).
5. **Testing the model** to ensure proper initialization.
6. **Using the model within ULDetection**.

By following these steps, your model will be fully integrated and available for use in UniversaLabeler.

---

## Next Steps
- **[Architecture Overview](architecture.md)** – Learn about the core system design.
- **[Advanced pipelines](advanced-usage-pipelines.md)** – Create pipelines with custom configurations.

**Congratulations! You have successfully added a new model to UniversaLabeler.** 🚀

