import os
import numpy as np
from ModelsFactory.image_caption.gpt_workspace.gpt_image_caption_model import GptImageCaption
from UL.ul_detection import ULDetection
from UL.ul_segmentation import ULSegmentation
from common.model_name_registry import MOST_CONFIDENCE, ModelNameRegistryDetection, ModelNameRegistrySegmentation

# Paths and Configurations
image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/tested_image/detection/street.png"
segmentation_directory = "output1/segmentation"
detection_directory = "output1/detection_directory"
gpt_api_key = "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"

# Step 1: Initialize GPT for Class Discovery
print("Step 1: Using GPT to discover all classes in the image...")
gpt_model = GptImageCaption(api_key=gpt_api_key)
gpt_model.init_model()
gpt_model.set_image(image_path)
detection_classes = gpt_model.set_prompt_get_all_classes_from_image(add_synonym=True)

# Step 2: Detect Objects with ULDetection
print("Step 2: Detecting objects using ULDetection...")
trex_input_class_bbox = {class_name: MOST_CONFIDENCE for class_name in detection_classes}


ul_detection = ULDetection(
    image_input=image_path,
    detection_class=detection_classes,
    use_nms=True,
    model_names=[
        ModelNameRegistryDetection.YOLO_WORLD.value,
        ModelNameRegistryDetection.DINOX_DETECTION.value,
        ModelNameRegistryDetection.TREX2.value,
        ModelNameRegistryDetection.DINO.value,
    ],
    trex_input_class_bbox=trex_input_class_bbox,
    filter_unwanted_classes=True,
)

nms_results, individual_results = ul_detection.process_image()
print("Detection Results:", nms_results)
# Save Detection Results
ul_detection.save_results(individual_results, nms_results, detection_directory)

# Step 3: Segmentation on Detected Bounding Boxes using ULSegmentation
print("Step 3: Applying segmentation to detected bounding boxes...")
bounding_boxes = [np.array(bbox) for bbox in nms_results["bboxes"]]
segmentation_classes = detection_classes

ul_segmentation = ULSegmentation(
    image_input=image_path,
    segmentation_class=segmentation_classes,
    sam2_predict_on_bbox=bounding_boxes,
    model_names=[
        ModelNameRegistrySegmentation.SAM2.value,
        ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value,
        ModelNameRegistrySegmentation.SEEM.value,
    ],

)

segmentation_results, model_results = ul_segmentation.process_image()
print("Segmentation Results:", segmentation_results)

# Save Segmentation Results
ul_segmentation.save_results(model_results, segmentation_directory)
