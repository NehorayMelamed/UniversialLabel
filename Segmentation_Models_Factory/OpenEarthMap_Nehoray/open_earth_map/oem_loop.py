import torch
import open_earth_map as oem
import os
import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path


def load_model(model_dir, model_name="u_former_best.pth", device="cuda"):
    """
    Loads the pre-trained UNetFormer model.

    Parameters:
    - model_dir (str): Directory where the model checkpoint is stored.
    - model_name (str): Name of the model checkpoint file.
    - device (str): Device to load the model onto ("cuda" or "cpu").

    Returns:
    - torch.nn.Module: Loaded model ready for inference.
    """
    network = oem.networks.UNetFormer(in_channels=3, n_classes=6)
    network = oem.utils.load_checkpoint(
        network,
        model_checkpoint_name=model_name,
        model_dir=model_dir,
    )
    return network.to(device)


def segment_image(image_path, model, device="cuda", save_result=False, output_root_dir="output"):
    """
    Segments an input image using a pre-trained UNetFormer model.

    Parameters:
    - image_path (str): Path to the input image (e.g., .tif, .png, .jpeg).
    - model (torch.nn.Module): Pre-trained segmentation model (UNetFormer).
    - device (str): Device to perform computation on ("cuda" or "cpu").
    - save_result (bool): If True, saves the segmented mask and overlayed image.
    - output_root_dir (str): Root directory to save the results.

    Returns:
    - np.ndarray: Segmented mask as a NumPy array.
    """
    # Get the base name (without extension) of the image to create the output directory
    base_name = Path(image_path).stem
    output_dir = os.path.join(output_root_dir, base_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    if image_path.endswith('.tif'):
        with rasterio.open(image_path, "r") as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
    else:
        img = cv2.imread(image_path)

    # Normalize and prepare the image for the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prd = model(img).squeeze(0).cpu()
    segmented_mask = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))

    if save_result:
        # Save the input image to the output directory
        input_image_filename = os.path.join(output_dir, f"{base_name}_input.png")
        original_image = plt.imread(image_path)
        plt.imsave(input_image_filename, original_image)

        # Save segmented mask
        mask_filename = os.path.join(output_dir, f"{base_name}_segmented.png")
        plt.imsave(mask_filename, segmented_mask)

        # Create and save overlayed image
        added_image = cv2.addWeighted(original_image, 0.6, segmented_mask, 0.4, 0)
        overlay_filename = os.path.join(output_dir, f"{base_name}_overlayed.png")
        plt.imsave(overlay_filename, added_image)

    return segmented_mask


if __name__ == "__main__":
    # Set directories
    model_dir = "/home/nehoray/PycharmProjects/Segmentation_Models_Factory/checkpoints/openEarhMap"
    image_path = "/home/nehoray/PycharmProjects/Segmentation_Models_Factory/data/images/images/daressalaam_102.tif"
    output_root_dir = "predictions"  # Root directory where each image's results will be saved

    # Load the model
    model = load_model(model_dir, device="cuda")

    # Segment the image and save results in an output directory named after the image
    segmented_mask = segment_image(image_path, model, device="cuda:0", save_result=True,
                                   output_root_dir=output_root_dir)

    print(f"Segmentation completed. Results saved in '{output_root_dir}/{Path(image_path).stem}'.")
