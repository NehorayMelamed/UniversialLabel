####################### IMPORTANT #######################
# This script demonstrates how to use the SEEM model for
# regular and text-based inference on images. The script
# loads the model and performs regular inference on a
# test image, then runs text-based inference on the same
# image using a text prompt. The script also shows how to
# use NumPy arrays as input for inference.
# 
# This script should be run from the father directory of the SEEM directory
# i.e, assuming ./SEEM is the directory containing the SEEM package,
# this script should be run from the directory containing ./SEEM (.example_usage.py).
########################################################

from SEEM.seem_usage import draw_regular_inference, draw_text_based_results, infer_image_text_prompt, load_model, infer_image
from PIL import Image
from SEEM.utils.visualizer import Visualizer
import numpy as np

# Constants
CONFIG_FILE_PATH = "/raid/Ben/a_checkpoints/SEEM/configs/seem/focall_unicl_lang_v1.yaml"
MODEL_CHECKPOINT_PATH = "/raid/Ben/a_checkpoints/SEEM/seem_focall_v1.pt"
IMAGE_PATH = "/raid/Ben/a_datasets/other/test_data/test.jpg"
REF_IMAGE_PATH = "/raid/Ben/a_datasets/other/test_data/ref.jpg"

# Usage example
model, all_classes, colors_list = load_model(CONFIG_FILE_PATH, MODEL_CHECKPOINT_PATH)  # Load the model and class information

image = Image.open(IMAGE_PATH)  # Open the test image
ref_image = Image.open(REF_IMAGE_PATH)  # Open the reference image

# Regular inference (Panoptic, Semantic, and Instance Segmentation)
results, image_np = infer_image(model, image)  # Run regular inference
visual = Visualizer(image_np, model.model.metadata)  # Create a visualizer object for the image

# Panoptic segmentation result
result_image_panoptic = draw_regular_inference(visual, results, seg_type='panoptic')  # Draw panoptic segmentation result
result_image_panoptic.save("result_panoptic.png")  # Save the panoptic segmentation result

# Semantic segmentation result
result_image_semantic = draw_regular_inference(visual, results, seg_type='semantic')  # Draw semantic segmentation result
result_image_semantic.save("result_semantic.png")  # Save the semantic segmentation result

# Instance segmentation result
result_image_instance = draw_regular_inference(visual, results, seg_type='instance')  # Draw instance segmentation result
result_image_instance.save("result_instance.png")  # Save the instance segmentation result

# Text-based inference with grounding
image_input = {"image": image, "mask": np.ones_like(np.array(image))}  # Prepare the input for text-based inference
pred_masks_pos, image_ori, pred_class = infer_image_text_prompt(model, image_input, reftxt="the white van")  # Run text-based inference
visual_text = Visualizer(image_ori, model.model.metadata)  # Create a visualizer for the original image
result_image_text = draw_text_based_results(visual_text, [pred_masks_pos], "the white van", pred_class, colors_list)  # Draw the text-based result
result_image_text.save("result_text_task.png")  # Save the text-based result


# If using NumPy array as input (instead of PIL image)
numpy_image = np.array(image)  # Convert the first image to NumPy array for demonstration
results_numpy, image_np_from_numpy = infer_image(model, numpy_image)  # Run inference using the NumPy array
visual_numpy = Visualizer(image_np_from_numpy, model.model.metadata)  # Create a visualizer for the NumPy image

# Panoptic segmentation for NumPy image
result_image_panoptic_numpy = draw_regular_inference(visual_numpy, results_numpy, seg_type='panoptic')  # Panoptic segmentation for NumPy image
result_image_panoptic_numpy.save("result_panoptic_numpy.png")  # Save the result

# Text-based inference on NumPy image
numpy_image_input = {"image": numpy_image, "mask": np.ones_like(numpy_image)}  # Prepare NumPy input for text-based inference
pred_masks_pos_numpy, image_ori_numpy, pred_class_numpy = infer_image_text_prompt(model, numpy_image_input, reftxt="the white van")  # Text-based inference for NumPy image
visual_text_numpy = Visualizer(image_ori_numpy, model.model.metadata)  # Visualizer for NumPy-based image
result_image_text_numpy = draw_text_based_results(visual_text_numpy, [pred_masks_pos_numpy], "the white van", pred_class_numpy, colors_list)  # Draw result for NumPy image
result_image_text_numpy.save("result_text_task_numpy.png")  # Save the result for NumPy input