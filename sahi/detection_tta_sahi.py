import os
from typing import Callable, List, Tuple
import cv2
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential  # For applying augmentation sequences to images
from kornia import augmentation as aug
from tqdm import tqdm
from ultralytics import YOLO, YOLOWorld
from PIL import Image

from ModelsFactory.Detection.detection_base_model import DetectionBaseModel
from sahi.sahi_utils import adjust_bboxes, calculate_slice_bboxes, crop_numpy_batch, crop_tensor, crop_torch_batch, imshow_dgx, pad_to_dimensions, scale_frame


def inverse_transformations(aug_sequence: AugmentationSequential,
                            augmented_crop: torch.Tensor,
                            slice_predictions_BB: np.ndarray) -> torch.Tensor:
    """
    Apply inverse transformations to the augmented image slice and adjust the predictions.

    Args:
        aug_sequence (AugmentationSequential): Kornia augmentation sequence to apply.
        augmented_crop (torch.Tensor): Tensor of shape (1, C, H, W) containing the augmented image slice.
        slice_predictions_BB (np.ndarray): Array of shape (N, 6) containing slice predictions with bounding box coordinates and class confidence.

    Returns:
        torch.Tensor: Inversely transformed keypoints for the predictions.
    """
    ### Extract Keypoints From BB Predictions: ###
    predicted_keypoints = torch.tensor(
        np.concatenate((
            slice_predictions_BB[:, 0:2],  # Extract x0y0 coordinates
            slice_predictions_BB[:, 1:3][:, ::-1],  # Extract and reverse x1y0 coordinates
            slice_predictions_BB[:, 0:4:3],  # Extract x0y1 coordinates
            slice_predictions_BB[:, 2:4],  # Extract x1y1 coordinates
        ))
    )[None, ...]  # Combine keypoints and reshape

    ### Apply inverse augmentations to the keypoints: ###
    _inversed_image_crop, inversed_keypoints = aug_sequence.inverse(augmented_crop,
                                                                     predicted_keypoints.float())  # Apply inverse augmentations
    return inversed_keypoints  # Return inversed keypoints

def image_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a NumPy array.
    Args:
        image_tensor (torch.Tensor): The input image tensor as a torch tensor. (C, H, W), normalized between [0, 1].

    Returns:
        np.ndarray: The converted image as a NumPy array. (H, W, C), normalized between [0, 1].
    """
    return image_tensor.permute(1, 2, 0).cpu().numpy()

def image_numpy_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy array to a torch tensor.
    Args:
        image_np (np.ndarray): The input image as a NumPy array. (H, W, C), normalized between [0, 1].

    Returns:
        torch.Tensor: The converted image as a torch tensor. (C, H, W), normalized between [0, 1].
    """
    return torch.tensor(image_np).permute(2, 0, 1).float()

def show_boxes_on_image_dgx(image: np.ndarray, boxes: np.ndarray):
    """
    Display the image with bounding boxes.
    Args:
        image (np.ndarray): The input image as a NumPy array. normalized, [0,1], (H, W, C), float32.
        boxes (np.ndarray): Array of shape (N, 4) containing bounding box coordinates. xyxy, not normalized.
    """
    output_image = (image * 255).astype(np.uint8).copy()

    # Iterate over the predictions and draw bounding boxes with labels
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        # Draw the bounding box
        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    imshow_dgx(output_image)


def draw_bounding_boxes_on_image(image: np.ndarray, bbx_np: np.ndarray):
    """
    Display the image with bounding boxes and labels.

    Args:
        image (np.ndarray): The input image as a NumPy array. normalized, [0, 1], (H, W, C), float32.
        bbx_np (np.ndarray): Array of shape (N, 4) or (N, 6) containing slice predictions with bounding box coordinates, class id, and class confidence.
                             If only coordinates are present, the array shape is (N, 4). If class id and confidence are also present, the shape is (N, 6).
    """
    output_image = (image * 255).astype(np.uint8).copy()

    # Iterate over the predictions and draw bounding boxes with labels
    for prediction in bbx_np:
        if len(prediction) >= 4:
            # Unpack bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, prediction[:4])

            # Draw the bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if len(prediction) >= 6:
                # If class id and confidence score are available, draw the label
                cls_id = int(prediction[4])
                score = float(prediction[5])

                # Prepare the label with class and confidence
                label = f"Class: {cls_id} Score: {score:.3f}"

                # Get the label size for background padding
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                top_left = (x_min, y_min - label_height - baseline)
                bottom_right = (x_min + label_width, y_min)

                # Draw background for the label
                cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), cv2.FILLED)

                # Draw the label on top of the background rectangle
                cv2.putText(output_image, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output_image


def show_bbx_on_image_dgx(image:np.ndarray, bbx_np:np.ndarray):
    output_image = draw_bounding_boxes_on_image(image, bbx_np)

    imshow_dgx(output_image)
    
def save_detection_image_to_disk(image: np.ndarray, bbx_np: np.ndarray, save_path: str):
    """
    Save the image with bounding boxes and labels to disk.
    Args:
        image (np.ndarray): The input image as a NumPy array. normalized, [0,1], (H, W, C), float32.
        bbx_np (np.ndarray): Array of shape (N, 6) containing slice predictions with bounding box coordinates and class confidence. xyxy, not normalized.
        save_path (str): The path to save the image with bounding boxes and labels.
    """
    output_image = draw_bounding_boxes_on_image(image, bbx_np)

    cv2.imwrite(save_path, output_image)

def merge_keypoints_to_bbox(slice_predictions_BB: np.ndarray, inversed_keypoints: torch.Tensor) -> np.ndarray:
    """
    Convert predicted keypoints back to bounding box format after inverse transformations.

    Args:
        slice_predictions_BB (np.ndarray): Array of shape (N, 6) containing slice predictions with bounding box coordinates and class confidence.
        inversed_keypoints (torch.Tensor): Tensor of shape (1, N, 8, 2) containing the inversely transformed keypoints.

    Returns:
        np.ndarray: Updated slice predictions with corrected bounding box coordinates.
    """
    numpy_keypoints = inversed_keypoints.numpy().squeeze()  # Convert keypoints to numpy and squeeze
    number_of_predictions = len(slice_predictions_BB)  # Get the number of predictions
    for i in range(number_of_predictions):  # Iterate over each prediction
        bbox = numpy_keypoints[i::number_of_predictions]  # Extract bounding box keypoints
        x0y0 = np.min(bbox, axis=0)  # Get minimum x0y0 coordinates
        x1y1 = np.max(bbox, axis=0)  # Get maximum x1y1 coordinates
        slice_predictions_BB[i, :4] = np.concatenate(
            (x0y0, x1y1))  # Update slice predictions with bounding box coordinates
    return slice_predictions_BB  # Return updated predictions

def process_yolo_bbx(yolo_result, detection_conf_threshold=0.5):
    to_ret = []
    for bbox, cls, score in zip(yolo_result.boxes.xyxy.cpu().numpy(), yolo_result.boxes.cls.cpu().numpy(), yolo_result.boxes.conf.cpu().numpy()):
        if score >= detection_conf_threshold:
            x1, y1, x2, y2 = map(int, bbox)
            to_ret.append((x1, y1, x2, y2, cls, score))
    return np.array(to_ret)

# YOLO inference callback for detection
def yolo_inference_callback(yolo_model: DetectionBaseModel, images_tensor: torch.Tensor,
                            detection_conf_threshold: float = 0.5, **kwargs) -> List[np.ndarray]:
    """
    Callback function for running YOLO object detection on a batch of images.

    Args:
        yolo_model (DetectionBaseModel): The YOLO model instance to perform object detection.
        images_tensor (torch.Tensor): The input image tensor as a torch tensor, normalized between [0, 1]. (B, C, H, W)
        detection_conf_threshold (float, optional): Detection confidence threshold. Defaults to 0.5.
        **kwargs (dict): Additional keyword arguments to pass to the inference function.

    Returns:
        List[np.ndarray]: List of numpy arrays containing the bounding box coordinates, class IDs, and confidence scores. List of length B, each element is an array of shape (N, 6).
    """
    # Convert tensor back to a NumPy array with the appropriate shape and scale to [0, 255]
    images_np = (images_tensor.cpu().numpy() * 255).astype(np.uint8)  # (B, C, H, W) -> [0, 255]
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)

    final_results = []

    # Process each image separately
    for image_np in images_np:
        yolo_model.set_image(image_np)  # Set the image for inference
        results = yolo_model.get_result()  # Perform inference, expect a list of results

        # Make sure results is iterable and process accordingly
        for result in results:
            if hasattr(result, 'boxes'):
                final_results.append(process_yolo_bbx(result, detection_conf_threshold))
            else:
                print("Warning: Inference result does not have 'boxes' attribute.")

    return final_results


def apply_augmentations(
    image_tensor: torch.Tensor,
    aug_sequence: AugmentationSequential
) -> np.ndarray:
    """
    Apply augmentations to the image tensor.

    Inputs:
    - image_tensor (torch.Tensor):
        - The input image slice as a tensor.
        - Shape: (1, C, H, W), where C is the number of channels, H is the height, and W is the width.

    - aug_sequence (AugmentationSequential):
        - Kornia augmentation sequence to apply to the image tensor.

    Outputs:
    - augmented_image_tensor (torch.Tensor):
        - The augmented image tensor.
        - Shape: (1, C, H, W), where C is the number of channels, H is the height, and W is the width.
    """
    ### HANDLE EMPTY INPUT CASES: ###
    if image_tensor is None or image_tensor.numel() == 0:
        raise ValueError("The input image tensor is empty or None.")  # Raise an error if the input image tensor is empty

    ### INITIALIZE CLAHE PARAMETERS IF PRESENT IN AUGMENTATIONS: ###
    flag_clahe_in_augmentations = False  # Flag to check if CLAHE is in augmentations
    for current_aug in aug_sequence:  # Loop through augmentations
        if 'RandomClahe' in current_aug.__str__():  # Check if RandomClahe is in augmentations
            flag_clahe_in_augmentations = True  # Set flag to True if CLAHE is found
            clahe_grid_size = current_aug.flags['grid_size']  # Get grid size for CLAHE
            H_clahe, W_clahe = clahe_grid_size  # Unpack CLAHE grid size
            H_clahe = 1 * H_clahe  # Update CLAHE grid height
            W_clahe = 1 * W_clahe  # Update CLAHE grid width

    ### ADJUST IMAGE TENSOR SIZE FOR CLAHE IF NECESSARY: ###
    if flag_clahe_in_augmentations:
        H, W = image_tensor.shape[-2:]  # Get the height and width of the image tensor
        original_size = image_tensor.shape[-2:]  # Store the original size for later use
        H_new = H + (H_clahe - H % H_clahe)  # Adjust height to be a multiple of the CLAHE grid size
        W_new = W + (W_clahe - W % W_clahe)  # Adjust width to be a multiple of the CLAHE grid size
        image_tensor = crop_tensor(image_tensor, (H_new, W_new))  # Crop tensor to the new size

    ### APPLY AUGMENTATIONS TO IMAGE TENSOR: ###
    augmented_crop, _augmented_keypoints = aug_sequence(
        image_tensor.contiguous(),  # Normalize the image tensor by dividing by 255
        torch.tensor([[[200, 200]]]).float()  # Dummy keypoints tensor, necessary for some augmentations
    )  # Apply augmentations to the image tensor

    ### CROP AUGMENTED IMAGE BACK TO ORIGINAL SIZE IF NECESSARY: ###
    if flag_clahe_in_augmentations:
        augmented_crop = crop_tensor(augmented_crop, original_size)  # Crop back to the original size

    return augmented_crop # return the augmented image tensor

def slice_aided_detection_inference(
        image: np.ndarray,
        model,
        model_input_dimensions: Tuple[int, int],
        slice_dimensions: Tuple[int, int],
        detection_conf_threshold: float,
        inference_callback,
        transforms: list = None,
        zoom_factor: float = 1.0,
        required_overlap_height_ratio=0.2, 
        required_overlap_width_ratio=0.2,
        **detection_callback_kwargs
) -> np.ndarray:
    """
    Performs model inference on image slices, applying transformations and reversing them as needed.

    Args:
        image (np.ndarray): Input image array normalized to [0, 1] of shape (H, W, C).
        model: Pre-trained model for object detection.
        model_input_dimensions (Tuple[int, int]): Target (H, W) for resizing before inference.
        slice_dimensions (Tuple[int, int]): Height and width of each image slice for processing.
        detection_conf_threshold (float): Confidence threshold for detections.
        inference_callback (Callable): Callback function for running inference on each slice, should receive: (model, torch.tensor (B,C,H,W), **kwrgs), returns List of numpy arrays containing the bounding box coordinates, class IDs, and confidence scores. list of length B, each element is an array of shape (N, 6).
        zoom_factor (float): Scaling factor for the image.
        transforms (list, optional): list of transformations to apply.

    Returns:
        np.ndarray: Array of detections with coordinates, classes, and scores.
    """
    predictions = np.empty((0, 6))  # Initialize an empty array for predictions
    aug_sequence = AugmentationSequential(*transforms, data_keys=["input", "keypoints"])

    # Step 1: Scale the frame to the model's input dimensions
    original_image_dimensions = image.shape[0:2]
    
    scaled_image = image.copy()
    if zoom_factor and zoom_factor != 1.0:
        scaled_image, zoom_factor = scale_frame(image, zoom_factor=zoom_factor)
    
    # Step 3: Calculate slice bounding boxes to divide the image into slices
    scaled_image_dimensions = scaled_image.shape[:2]
    scaled_image_height, scaled_image_width = scaled_image_dimensions
    slice_bboxes = [[0, 0, scaled_image_width, scaled_image_height]]
    
    assert slice_dimensions<=(scaled_image_height, scaled_image_width), "Slice dimensions are larger than the scaled image dimensions"
    
    if slice_dimensions != (0, 0) and slice_dimensions != (scaled_image_height, scaled_image_width):
        slice_bboxes = calculate_slice_bboxes(scaled_image_dimensions, slice_dimensions, required_overlap_height_ratio, required_overlap_width_ratio)
    
    for idx, (x0, y0, x1, y1) in enumerate(slice_bboxes):
        # crop slice of the image
        slice = scaled_image[y0:y1, x0:x1] # slice is a numpy array, (slice_H, slice_W, C)
        slice_dimensions = slice.shape[:2] # (slice_H, slice_W)
        
        # scale scaled_image_slice for model input dimensions
        slice_scaled_for_model, _ = scale_frame(slice, new_dimensions=model_input_dimensions) # slice_scaled_for_model is a numpy array, (model_input_H, model_input_W, C)
        slice_scaled_for_model_dimensions = slice_scaled_for_model.shape[:2] # (model_input_H, model_input_W)
        
        # switch to tensor
        slice_scaled_for_model_tensor = torch.tensor(slice_scaled_for_model).permute(2, 0, 1).unsqueeze(0).float() # 1,C,H,W
        
        # augment the slice
        augmented_slice_scaled_for_model_tensor = apply_augmentations(slice_scaled_for_model_tensor, aug_sequence) # 1,C,H,W
        
        # apply inference
        augmented_slice_scaled_for_model_predictions = inference_callback(model, augmented_slice_scaled_for_model_tensor, detection_conf_threshold, **detection_callback_kwargs)[0]
        
        # show results on augmented slice
        # show_bbx_on_image_dgx(image_tensor_to_numpy(augmented_slice_scaled_for_model_tensor[0]), augmented_slice_scaled_for_model_predictions)
        
        if augmented_slice_scaled_for_model_predictions.shape[0] > 0:
            # inverse transformations
            inversed_keypoints = inverse_transformations(aug_sequence, slice_scaled_for_model_tensor, augmented_slice_scaled_for_model_predictions)
            slice_scaled_for_model_predictions = merge_keypoints_to_bbox(augmented_slice_scaled_for_model_predictions, inversed_keypoints)
            
            # show results on slice scaled for model
            # show_bbx_on_image_dgx(slice_scaled_for_model, slice_scaled_for_model_predictions)
            
            # rescale slice_predictions from model dimension to slice dimension and offset to match original image
            final_slice_predictions_with_offset = adjust_bboxes(slice_scaled_for_model_predictions, slice_scaled_for_model_dimensions, slice_dimensions, x0, y0)
            
            # show current results on scaled_image
            # show_bbx_on_image_dgx(scaled_image, final_slice_predictions_with_offset)
            
            # append to predictions
            predictions = np.vstack((predictions, final_slice_predictions_with_offset))
        
    # rescale all the boundign boxes to the original image size
    predictions = adjust_bboxes(predictions, scaled_image_dimensions, original_image_dimensions)
        
    return predictions


if __name__ == '__main__':

    # Load YOLO model and perform detection on the entire image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model_path = "/home/nehoray/PycharmProjects/UniversaLabeler/common/weights/yolov8s-world.pt"

    transforms_list = [
            aug.RandomHorizontalFlip(p=0.5),
            # aug.RandomEqualize(p=0.5),
            # aug.RandomRotation(degrees=90, p=1),
            # aug.RandomVerticalFlip(p=0.5),
            # aug.RandomChannelShuffle(p=0.5)
        ]

    folder_path = '/home/nehoray/PycharmProjects/UniversaLabeler/sahi/input'
    images = os.listdir(folder_path)

    yolo_model_path = '/home/nehoray/PycharmProjects/UniversaLabeler/common/weights/yolov8s-world.pt'
    yolo_model = YOLOWorld(yolo_model_path)
    yolo_model.set_classes(["car", "bus"])
    save_folder = '/home/nehoray/PycharmProjects/UniversaLabeler/sahi/results'
    os.makedirs(save_folder, exist_ok=True)
    for image_filename in tqdm(images):
        image_path = os.path.join(folder_path, image_filename)
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image) / 255.0
        H,W = image_np.shape[:2]

        predictions = slice_aided_detection_inference(
            image=image_np,                    # The input image as a NumPy array, normalized to a range of 0 to 1.
            model=yolo_model,             # The YOLO model instance to perform object detection.
            model_input_dimensions=(640, 640), # Target dimensions (height, width) to resize the image or slices before inference.
            slice_dimensions=(int(H), int(W)),       # Size (height, width) for each slice when dividing the image. Set to (0, 0) to process the full image at once.
            detection_conf_threshold=0.8,      # Confidence threshold for filtering out low-confidence detections.
            inference_callback=yolo_inference_callback,  # Callback function for running inference on each slice.
            transforms=transforms_list,                   # Optional list of transformations (e.g., augmentations) to apply to each slice before inference.
            zoom_factor=1.0,               # If True, resizes the image to `model_input_dimensions`; if False, uses `zoom_factor` instead.
            required_overlap_height_ratio=0.0,
            required_overlap_width_ratio=0.0
        )

        image = Image.open(image_path)
        image_np = np.asarray(image) / 255.0
        # convert to RGB to BGR suing cv2
        image_np = image_np[..., ::-1]
        save_path = os.path.join(save_folder, image_filename)
        save_detection_image_to_disk(image_np, bbx_np=predictions, save_path=save_path)