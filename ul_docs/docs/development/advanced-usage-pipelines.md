# Advanced Pipelines in UniversaLabeler

## Introduction

UniversaLabeler allows for the creation of highly customizable and modular pipelines. By combining different models, configurations, and post-processing techniques, users can build advanced workflows tailored to their specific needs. The system enables integration across multiple deep learning models and supports flexible configurations for optimal results.

This guide will provide examples of how to design and implement complex pipelines using UniversaLabeler.

---

## Designing Custom Pipelines

Creating an advanced pipeline involves:
- Selecting and configuring the appropriate detection, segmentation, and captioning models.
- Integrating preprocessing and post-processing layers.
- Utilizing multi-model strategies to improve accuracy.
- Combining information from different sources to refine results.

### Example Use Cases

Here are some examples of advanced pipelines implemented using UniversaLabeler.

### **Pipeline 1: Building Front Analysis**

This pipeline analyzes building fronts using multiple detection and segmentation models, refining results using hierarchical logic.

#### **Steps:**
1. **Extract all objects from the image** using a GPT-based captioning model.
2. **Expand object classes** by enriching them using a language model.
3. **Run detection models:**
   - `YOLO_WORLD`
   - `DINOX_DETECTION`
   - `OPENGEOS`
4. **Filter and refine bounding boxes** using the Non-Maximum Suppression (NMS) algorithm.
5. **Feed the most confident detections to TREX** for tracking and reference-based detection.
   ```python
   trex_input_class_bbox = {class_name: MOST_CONFIDENCE for class_name in detection_classes}
   ```
6. **Apply segmentation models:**
   - `DINOX_SEGMENTATION`
   - `SEEM`
7. **Define model priority order:**
   ```python
   model_priorities = {
       ModelNameRegistrySegmentation.SEEM.value: 5,
       ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value: 4,
   }
   ```
8. **Generate bounding boxes for SAM2 to refine segmentations:**
   ```python
   bounding_boxes = [np.array(bbox) for bbox in nms_results["bboxes"]]
   ```
9. **Save and visualize results.**

📍 **Example File:**
📂 `UL/example_usage/front_building/dinox_opengeos_trex.py`

---

### **Pipeline 2: Street from Sky**

This pipeline processes aerial images of streets, extracting infrastructure details with both segmentation and detection models.

#### **Steps:**
1. **Detect objects in the image** using:
   - `YOLO_WORLD`
   - `DINOX_DETECTION`
   - `OPENGEOS`
2. **Apply SAHI preprocessing** for enhanced detections on high-resolution images.
3. **Use Segmentation Models:**
   - `DINOX_SEGMENTATION`
   - `SAM`
4. **Perform post-processing with SegSelector** to merge segmentation masks effectively.
5. **Output structured metadata and visual results.**

📍 **Example File:**
📂 `UL/example_usage/street_from_sky/example_1.py`

---

### **Pipeline 3: Front View of Street**

This pipeline processes street-level images, performing detailed object detection and classification.

#### **Steps:**
1. **Identify all major street objects** using:
   - `YOLO_WORLD`
   - `DINOX_DETECTION`
2. **Perform segmentation on detected objects** using:
   - `SEEM`
   - `DINOX_SEGMENTATION`
3. **Classify detected objects using an image captioning model.**
4. **Refine detections with bounding boxes for TREX tracking.**
5. **Save results and overlay visualized predictions on the image.**

📍 **Example File:**
📂 `UL/example_usage/street_front_view/example1.py`

---

## Exploring More Pipelines

The UniversaLabeler repository contains multiple ready-to-use pipeline configurations.

🔍 **Check out additional examples:**
📂 `UL/example_usage/`

Users are encouraged to experiment and create their own pipelines by modifying configurations, integrating new models, and combining detection, segmentation, and captioning tasks in innovative ways.

Happy experimenting! 🚀

