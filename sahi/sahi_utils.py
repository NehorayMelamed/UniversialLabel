from typing import Callable, List, Tuple
import cv2
import numpy as np
import albumentations as albu
import torch
from kornia.augmentation import AugmentationSequential  # For applying augmentation sequences to images
import kornia as K
from PIL import Image
from ultralytics import YOLO

import time
import torch
from torchvision.transforms import ToPILImage
from PIL import Image

def imshow_dgx(image_input, new_size=None, filename='display.png'):
    """
    Save a given tensor or numpy array as an image file.

    Args:
        image_input (torch.Tensor or np.ndarray): The image data to save as an image. 
                                                  For tensors, it should have the shape (C, H, W) or (H, W).
                                                  For numpy arrays, it should have the shape (H, W, C) or (H, W).
        filename (str): The path where the image will be saved.
        new_size (tuple or list, optional): New size (H, W) to resize the image before saving. Defaults to None.
    """
    # Check if the input is a torch tensor
    if isinstance(image_input, torch.Tensor):
        # Check if the tensor has a batch dimension
        if image_input.dim() == 4:
            # Remove the batch dimension
            image_input = image_input.squeeze(0)
        
        # Ensure the tensor has the shape (C, H, W)
        if image_input.dim() == 2:  # (H, W)
            image_input = image_input.unsqueeze(0)  # Add a channel dimension

        if image_input.dim() == 3 and image_input.size(0) == 1:  # (1, H, W)
            image_input = image_input.repeat(3, 1, 1)  # Convert to 3-channel (grayscale to RGB)

        # Convert the tensor to a PIL image
        to_pil_image = ToPILImage()
        image = to_pil_image(image_input)

    # Check if the input is a numpy array
    elif isinstance(image_input, np.ndarray):
        # Ensure the numpy array has the shape (H, W, C)
        if image_input.ndim == 2:  # (H, W)
            image_input = np.expand_dims(image_input, axis=-1)  # Add a channel dimension

        if image_input.shape[2] == 1:  # (H, W, 1)
            image_input = np.repeat(image_input, 3, axis=2)  # Convert to 3-channel (grayscale to RGB)

        # Convert the numpy array to a PIL image
        image = Image.fromarray(image_input.astype(np.uint8))

    else:
        raise ValueError("Input must be a torch.Tensor or a numpy.ndarray")

    # Resize the image if new_size is provided
    if new_size is not None:
        new_size = [new_size[1], new_size[0]]
        image = image.resize(new_size)

    # Save the image
    image.save(filename)

def imshow_dgx_video(input_tensor, FPS, stride=1, new_size=None):
    """
    Display a sequence of images as a video by calling imshow_dgx on each frame.

    Args:
        input_tensor (list or np.ndarray or torch.Tensor): The input sequence of images. 
                                                           Can be a list of numpy arrays or torch tensors,
                                                           a numpy array of shape [T, H, W, C], or a torch tensor of shape [T, C, H, W].
        FPS (float): Frames per second for displaying the video.
        stride (int, optional): Number of frames to skip between displayed frames. Defaults to 1.
        new_size (tuple or list, optional): New size (H, W) to resize the images before displaying. Defaults to None.
    """
    def process_frame(frame):
        imshow_dgx(frame, new_size)
        time.sleep(1.0 / FPS)

    if isinstance(input_tensor, list):
        for i in range(0, len(input_tensor), stride):
            process_frame(input_tensor[i])
    elif isinstance(input_tensor, np.ndarray):
        for i in range(0, input_tensor.shape[0], stride):
            process_frame(input_tensor[i])
    elif isinstance(input_tensor, torch.Tensor):
        for i in range(0, input_tensor.size(0), stride):
            process_frame(input_tensor[i])
    else:
        raise ValueError("input_tensor must be a list, numpy array, or torch tensor")

def scale_frame(
        input_frame_numpy: np.ndarray,
        zoom_factor: float = 1.0,
        new_dimensions: Tuple[int, int] = (0, 0),
        size_multiplication_factor: int = 1,
        use_padding: bool = True
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Scale a frame by a zoom factor, resize it to new dimensions, or adjust it to ensure its dimensions are multiples of a given factor.

    Inputs:
    - input_frame_numpy (np.ndarray):
        - The input frame as a numpy array.
        - Shape: (H, W, C), where H is the height, W is the width, and C corresponds to the color channels.
    - zoom_factor (float):
        - The factor by which to scale the frame.
        - Default: 1.0 (no scaling).
    - new_dimensions (Tuple[int, int]):
        - The target dimensions as (height, width) for resizing the frame.
        - If set to (0, 0) or None, the frame will be scaled using `zoom_factor`.
        - Default: (0, 0).
    - size_multiplication_factor (int):
        - The factor by which the final dimensions must be divisible.
        - Default: 1 (no adjustment).
    - use_padding (bool):
        - If True, padding will be used to achieve dimensions as multiples of `size_multiplication_factor`.
        - If False, resizing will be used to achieve dimensions as multiples of `size_multiplication_factor`.
        - Default: True.

    Outputs:
    - scaled_frame_numpy (np.ndarray):
        - The scaled and adjusted frame as a numpy array.
        - Shape: (H_new, W_new, C).
    - zoom_factor_tuple (Tuple[float, float]):
        - The actual zoom factor applied as a tuple (zoom_factor_H, zoom_factor_W).
    """

    ### COPY INPUT FRAME TO PRESERVE ORIGINAL: ###
    scaled_frame_numpy = input_frame_numpy.copy()  # Create a copy of the input frame

    ### GET ORIGINAL FRAME DIMENSIONS: ###
    H_input_int, W_input_int, _ = scaled_frame_numpy.shape  # Extract height and width from the frame

    ### DETERMINE ZOOM FACTOR AND SCALE FRAME: ###
    if new_dimensions and new_dimensions != (0, 0):  # If valid new dimensions are provided
        H_new_int, W_new_int = new_dimensions  # Extract new height and width
        zoom_factor_tuple = (H_new_int / H_input_int, W_new_int / W_input_int)  # Calculate zoom factor
        scaled_frame_numpy = cv2.resize(scaled_frame_numpy,
                                        (int(W_new_int), int(H_new_int)))  # Resize frame to new dimensions
    else:  # If no new dimensions are provided, use zoom factor
        zoom_factor_tuple = (zoom_factor, zoom_factor)  # Initialize zoom factor tuple
        H_new_int = int(H_input_int * zoom_factor_tuple[0])  # Calculate new height
        W_new_int = int(W_input_int * zoom_factor_tuple[1])  # Calculate new width
        scaled_frame_numpy = cv2.resize(scaled_frame_numpy,
                                        (W_new_int, H_new_int))  # Resize frame according to zoom factor

    ### ADJUST TO SIZE MULTIPLICATION FACTOR: ###
    if size_multiplication_factor > 1:
        # Calculate required padding or resizing
        H_adjusted = (H_new_int + size_multiplication_factor - 1) // size_multiplication_factor * size_multiplication_factor
        W_adjusted = (W_new_int + size_multiplication_factor - 1) // size_multiplication_factor * size_multiplication_factor

        if use_padding:
            ### APPLY PADDING TO MEET SIZE MULTIPLICATION FACTOR: ###
            padding_H = H_adjusted - H_new_int
            padding_W = W_adjusted - W_new_int
            scaled_frame_numpy = np.pad(
                scaled_frame_numpy,
                [(0, padding_H), (0, padding_W), (0, 0)],
                mode='constant'
            )
        else:
            ### RESIZE TO MEET SIZE MULTIPLICATION FACTOR: ###
            scaled_frame_numpy = cv2.resize(
                scaled_frame_numpy,
                (W_adjusted, H_adjusted)
            )

    return scaled_frame_numpy, zoom_factor_tuple  # Return the scaled and adjusted frame and zoom factor


def pad_to_dimensions(
        input_frame_numpy: np.ndarray,
        target_dimensions_tuple: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
    """
    Zero-pad a given frame to the target dimensions. If target dimensions are None or (0, 0), return the original frame.

    Inputs:
    - input_frame_numpy (np.ndarray):
        - The input frame as a numpy array.
        - Shape: (H, W, C), where H is the height, W is the width, and C corresponds to the color channels.
    - target_dimensions_tuple (Tuple[int, int]):
        - The target dimensions as (height, width) for padding the frame.
        - If set to None or (0, 0), the original frame is returned unpadded.
        - Default: (0, 0).

    Outputs:
    - padded_frame_numpy (np.ndarray):
        - The padded frame as a numpy array.
        - Shape: (H_target, W_target, C), or the original frame shape if no padding is applied.
    """

    ### CHECK IF PADDING IS NECESSARY: ###
    if not target_dimensions_tuple or target_dimensions_tuple == (0, 0):  # If target dimensions are None or (0, 0)
        return input_frame_numpy  # Return the original frame unpadded

    ### EXTRACT ORIGINAL FRAME DIMENSIONS: ###
    H_input_int, W_input_int, _ = input_frame_numpy.shape  # Extract height and width from the frame

    ### PAD HEIGHT AND WIDTH IF NEEDED: ###
    H_target_int, W_target_int = target_dimensions_tuple  # Extract target height and width
    padded_frame_numpy = input_frame_numpy.copy()  # Create a copy of the input frame

    if H_target_int > H_input_int:  # Check if padding is needed for height
        padded_frame_numpy = np.pad(padded_frame_numpy, [(0, H_target_int - H_input_int), (0, 0), (0, 0)], mode='constant')

    if W_target_int > W_input_int:  # Check if padding is needed for width
        padded_frame_numpy = np.pad(padded_frame_numpy, [(0, 0), (0, W_target_int - W_input_int), (0, 0)], mode='constant')

    return padded_frame_numpy  # Return the padded frame

def calculate_slice_bboxes(
        image_dimensions: Tuple[int, int],
        slice_dimensions: Tuple[int, int] = (512, 512),
        required_overlap_height_ratio: float = 0.0,
        required_overlap_width_ratio: float = 0.0,
) -> List[List[int]]:
    """
    Calculates bounding boxes for dividing an image into slices with optional overlaps.
    Given the height and width of an image, calculates how to divide the image into slices according to the height and width provided. These slices are returned as bounding boxes in xyxy format.
    If required_overlap is bigger than 0, the slices will overlap. If it is 0, the slices will overlap only when needed to fill another slice.
    
    Args:
        image_dimensions (Tuple[int, int]): Dimensions of the image (height, width).
        slice_dimensions (Tuple[int, int]): Dimensions of each slice (height, width).
        required_overlap_height_ratio (float): Fractional overlap in height (e.g., 0.2 for 20% overlap).
        required_overlap_width_ratio (float): Fractional overlap in width (e.g., 0.2 for 20% overlap).
    
    Returns:
        List[List[int]]: A list of bounding boxes in xyxy format.
    """
    image_height, image_width = image_dimensions
    slice_height, slice_width = slice_dimensions

    # Calculate overlap in pixels based on required overlap ratios
    y_overlap = int(required_overlap_height_ratio * slice_height)
    x_overlap = int(required_overlap_width_ratio * slice_width)

    slice_bboxes = []

    # Loop to calculate all y-axis slice positions
    y_min = 0
    while y_min + slice_height <= image_height:
        y_max = y_min + slice_height

        x_min = 0
        # Loop to calculate all x-axis slice positions
        while x_min + slice_width <= image_width:
            x_max = x_min + slice_width

            # Add the bounding box
            slice_bboxes.append([x_min, y_min, x_max, y_max])

            # Move to the next x position
            x_min += slice_width - x_overlap

        # Handle last x slice if needed
        if x_min < image_width:
            x_min = image_width - slice_width
            x_max = image_width
            slice_bboxes.append([x_min, y_min, x_max, y_max])

        # Move to the next y position
        y_min += slice_height - y_overlap

    # Handle last y slice if needed
    if y_min < image_height:
        y_min = image_height - slice_height
        y_max = image_height

        x_min = 0
        # Loop to calculate all x-axis slice positions
        while x_min + slice_width <= image_width:
            x_max = x_min + slice_width

            # Add the bounding box
            slice_bboxes.append([x_min, y_min, x_max, y_max])

            # Move to the next x position
            x_min += slice_width - x_overlap

        # Handle last x slice if needed
        if x_min < image_width:
            x_min = image_width - slice_width
            x_max = image_width
            slice_bboxes.append([x_min, y_min, x_max, y_max])

    return slice_bboxes

def pad_torch_batch(input_tensor: torch.Tensor, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(t, pad_size):  # create a rigid shape for padding in dims CHW
            pad_start = np.floor(np.subtract(pad_size, torch.tensor(t.shape[-2:]).cpu().numpy()) / 2)
            pad_end = np.ceil(np.subtract(pad_size, torch.tensor(t.shape[-2:]).cpu().numpy()) / 2)
            return int(pad_start[1]), int(pad_end[1]), int(pad_start[0]), int(pad_end[0])

        return torch.nn.functional.pad(input_tensor, pad_shape(input_tensor, pad_size), mode='constant')
    else:
        return None

def pad_numpy_batch(input_arr: np.array, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape_HWC(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (0, 0)
        def pad_shape_HW(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1])

        if len(input_arr.shape) == 4:
            return np.array([np.pad(a, pad_shape_HWC(a, pad_size), 'constant', constant_values=0) for a in input_arr])
        elif len(input_arr.shape) == 3:
            return np.pad(input_arr, pad_shape_HWC(input_arr, pad_size), 'constant', constant_values=0)
        elif len(input_arr.shape) == 2:
            return np.pad(input_arr, pad_shape_HW(input_arr, pad_size), 'constant', constant_values=0)

    else:
        return None

def crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H, W = images.shape
    elif len(images.shape) == 3:
        H, W, C = images.shape #BW batch
    else:
        T, H, W, C = images.shape #RGB/NOT-BW batch

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) is list or type(crop_size_tuple_or_scalar) is tuple:
        cropW = crop_size_tuple_or_scalar[1]
        cropH = crop_size_tuple_or_scalar[0]
    else:
        cropW = crop_size_tuple_or_scalar
        cropH = crop_size_tuple_or_scalar
    cropW = min(cropW, W)
    cropH = min(cropH, H)

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###    imshow_torch(tensor1_crop)
    if len(images.shape) == 2:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)
    else:
        return pad_numpy_batch(images[:, start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)

def crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H,W = images.shape
        C = 1
    elif len(images.shape) == 3:
        C,H,W = images.shape
    elif len(images.shape) == 4:
        T, C, H, W = images.shape  # No Batch Dimension
    else:
        B, T, C, H, W = images.shape  # "Full" with Batch Dimension

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) is list or type(crop_size_tuple_or_scalar) is tuple:
        cropH = crop_size_tuple_or_scalar[0]
        cropW = crop_size_tuple_or_scalar[1]
    else:
        cropH = crop_size_tuple_or_scalar
        cropW = crop_size_tuple_or_scalar

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:  #center
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###
    if len(images.shape) == 2:
        return pad_torch_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_torch_batch(images[:, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 4:
        return pad_torch_batch(images[:, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    else:
        return pad_torch_batch(images[:, :, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    

def crop_tensor(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### crop_style = 'center', 'random', 'predetermined'
    if isinstance(images, torch.tensor):
        return crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W, flag_pad_if_needed)
    else:
        return crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W, flag_pad_if_needed)


def adjust_bboxes(
        original_bbs: np.ndarray,
        old_shape: tuple,
        new_shape: tuple,
        x0_offset: int = 0,
        y0_offset: int = 0,
) -> np.ndarray:
    """
    Adjust bounding boxes from scaled image to the original image dimensions, then update predictions.

    This function adjusts the bounding boxes (BB) based on the original and new dimensions of the image,
    considering uneven zoom factors if applicable.

    Args:
        original_bbs (np.ndarray): Array of shape (N, 6) where N is the number of slice predictions
                                           with bounding box coordinates in the format (x0, y0, x1, y1, class_id, confidence).
        old_shape (tuple):Dimensions of the image the new_predictions_bbs is of (bounding boxes are in the range of this dimension), (height, width).
        new_shape (tuple): New dimensions of the image (scaling the bouding boxes to this image) in the format (height, width).
        x0_offset (int): X-coordinate offset for the slice.
        y0_offset (int): Y-coordinate offset for the slice.

    Returns:
        np.ndarray: Updated predictions array in the format (x0, y0, x1, y1, class_id, confidence), where the  coordinates have been updated to the new dimensions.
    """
    bbs = original_bbs.copy()  # Copy the original bounding boxes
    
    ### Calculate Zoom Factors for Width and Height: ###
    zoom_factor_x = new_shape[1] / old_shape[1]  # Calculate zoom factor for width
    zoom_factor_y = new_shape[0] / old_shape[0]  # Calculate zoom factor for height

    ### Adjust Bounding Boxes to Scaled Frame: ###

    ### Adjust Bounding Boxes to Original Frame: ###
    bbs[:, 0] *= zoom_factor_x  # Scale x0 coordinate back to original frame dimensions
    bbs[:, 2] *= zoom_factor_x  # Scale x1 coordinate back to original frame dimensions
    bbs[:, 1] *= zoom_factor_y  # Scale y0 coordinate back to original frame dimensions
    bbs[:, 3] *= zoom_factor_y  # Scale y1 coordinate back to original frame dimensions
    
    bbs[:, :4:2] += x0_offset  # Adjust x coordinates by x0
    bbs[:, 1:4:2] += y0_offset  # Adjust y coordinates by y0

    return bbs  # Return updated predictions