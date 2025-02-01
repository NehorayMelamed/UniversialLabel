# Basic Usage

## Introduction

This section provides an overview of how to use **UniversaLabeler (UL)** and its core functionalities. After completing the installation and downloading the necessary models, you will be ready to utilize the powerful **UL Detection, UL Segmentation, and Image Captioning** components.

!!! warning "Caution"
    You are now using an advanced and powerful AI-driven labeling tool. Please ensure responsible and accurate usage to achieve the best results.

---

## UL Detection

**ULDetection** is responsible for detecting objects in images and returning bounding box (BBOX) results. The class supports multiple models, allowing users to process images using **prompt-based detection, predefined classes, and zero-shot learning models**.

### Data Classes
Several **data classes** are available under the `common` directory to manage object detection.

```python
class DetectionClassesName(Enum):
   SmallVehicle = "SmallVehicle"  # Alfred
   BigVehicle = "BigVehicle"      # Alfred
   # ... More classes
```

```python
class ModelNameRegistryDetection(Enum):
   DINO = "DINO"
   YOLO_WORLD = "YOLO_WORLD"
   # ... More models
```

---

### ULDetection Parameters

**ULDetection** accepts multiple parameters upon initialization:

- `image_input (str | np.ndarray)`: Path to the image or a NumPy array representation.
- `detection_class (List[str])`: List of detection class names or prompts.
- `class_priorities (Dict[str, int])`: Priorities for class selection during post-processing.
- `model_priorities (Dict[str, int])`: Priorities for model selection in case of overlapping detections.
- `use_nms (bool)`: Whether to apply **Non-Maximum Suppression (NMS)** to refine results.
- `nms_advanced_params (dict)`: Advanced NMS settings for customization.
- `filter_unwanted_classes (bool)`: If `True`, removes unwanted detected classes.
- `model_names (List[str])`: List of models to be used for detection.
- `trex_input_class_bbox (dict)`: Bounding box configuration for **TREX** reference-based detection.
- `sahi_models_params (dict)`: Configuration for **SAHI preprocessing algorithms**.

---

### Example: Creating an Instance of ULDetection

```python
image_path = "data/tested_image/sample.jpeg"

detection_classes = ["window", "car", "person"]

# SAHI Model Parameters
sahi_model_params = {
   ModelNameRegistryDetection.YOLO_WORLD.value: {
       'slice_dimensions': (256, 256),
       'detection_conf_threshold': 0.7
   }
}

ul_detection = ULDetection(
   image_input=image_path,
   detection_class=detection_classes,
   class_priorities={"window": 2, "car": 1},
   model_priorities={ModelNameRegistryDetection.YOLO_WORLD.value: 2, ModelNameRegistryDetection.DINOX_DETECTION.value: 1},
   use_nms=True,
   model_names=[ModelNameRegistryDetection.YOLO_WORLD.value, ModelNameRegistryDetection.DINOX_DETECTION.value],
   filter_unwanted_classes=True,
   sahi_models_params=sahi_model_params
)
```

---

### Processing an Image

```python
nms_results, individual_results = ul_detection.process_image()
```

- **`nms_results`**: The refined detection results after applying **NMS**.
- **`individual_results`**: Raw detection results from each individual model.

To **save the results**, run:

```python
output_directory = "./output_results"
ul_detection.save_results(individual_results, nms_results, output_directory)
```

---

## UL Segmentation

**ULSegmentation** performs pixel-wise segmentation of images using multiple models. It supports **semantic segmentation, instance segmentation, and panoptic segmentation**.

### Data Classes

```python
class ModelNameRegistrySegmentation(Enum):
   DINOX_SEGMENTATION = "DINOX_SEGMENTATION"
   SAM = "SAM"
   # ... More models
```

---

### ULSegmentation Parameters

- `image_input (str | np.ndarray)`: Path to the image or a NumPy array representation.
- `segmentation_class (List[str])`: List of segmentation class names.
- `class_priorities (Dict[str, int])`: Priorities for resolving overlapping segmentations.
- `model_priorities (Dict[str, int])`: Model selection priorities in case of multiple segmentations.
- `use_segselector (bool)`: Enables the **SegSelector** algorithm for result refinement.
- `seg_selector_advanced_params (dict)`: Advanced SegSelector settings.
- `sam2_predict_on_bbox (List[np.ndarray])`: Bounding boxes to pass to **SAM2** for segmentation.

---

### Example: Creating an Instance of ULSegmentation

```python
image_path = "data/street/sample.png"

segmentation_classes = ["car", "bus"]

bounding_boxes = [
   np.array([468, 157, 518, 203]),
   np.array([313, 138, 408, 256])
]

ul_segmentation = ULSegmentation(
   image_input=image_path,
   segmentation_class=segmentation_classes,
   model_names=[ModelNameRegistrySegmentation.SAM2.value, ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value],
   sam2_predict_on_bbox=bounding_boxes,
   model_priorities={ModelNameRegistrySegmentation.SEEM.value: 5, ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value: 4}
)
```

---

### Processing an Image

```python
formatted_result, individual_results = ul_segmentation.process_image()
```

- **`formatted_result`**: SegSelector-refined segmentation results.
- **`individual_results`**: Raw segmentation results from each model.

To **save segmentation results**, run:

```python
ul_segmentation.save_results(individual_results, "output_segmentation_results")
```

---

## Image Captioning

The **Image Captioning** module provides **image description, classification, and object analysis using large language models (LLMs)**.

### Example Functions

```python
# Check if an object is present in the image
does_exist = ul_caption.set_prompt_get_does_is("bus", threshold=0.30)
```

```python
# Generate a detailed image description
description = ul_caption.set_prompt_get_describe_image(detail=DetailLevel.HIGH)
```

```python
# Get all detected classes with synonyms and possible objects
classes = ul_caption.set_prompt_get_all_classes_from_image(also_like_to_be=True, add_synonym=True)
```

---

## Summary

This guide covers the **basic usage** of **UniversaLabeler** for:

- **UL Detection**: Object detection using bounding boxes.
- **UL Segmentation**: Pixel-wise image segmentation.
- **Image Captioning**: Image classification and description.

For more **advanced configurations**, check out the **[Advanced Usage](advanced-usage.md)** section.

---

## Next Steps

- **[Model Integration](../../development/model-integration.md)** – Extend with custom models.
- **[API Reference](../../api/api-reference.md)** – Explore API functionalities.

---

