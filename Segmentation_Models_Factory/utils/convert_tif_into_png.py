import os
import rasterio
import cv2
import numpy as np
from pathlib import Path


def convert_tif_to_png(input_dir, output_dir):
    """
    Converts all .tif images in the input directory to .png images and saves them to the output directory,
    preserving the correct color.

    Parameters:
    - input_dir (str): Directory containing .tif images.
    - output_dir (str): Directory to save the converted .png images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .tif files from the input directory
    tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

    for tif_file in tif_files:
        # Read the .tif image using rasterio
        tif_path = os.path.join(input_dir, tif_file)
        with rasterio.open(tif_path) as src:
            # Read all bands and normalize to 0-255
            img = src.read().astype(np.float32)  # Read all bands
            img_min = img.min()
            img_max = img.max()
            img = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)  # Normalize

            # If the image has more than 3 channels, only select the first 3 channels for RGB
            if img.shape[0] >= 3:
                img = img[:3, :, :].transpose(1, 2, 0)  # Convert to HWC (Height, Width, Channels)
            else:
                print(f"Skipping {tif_file}: Less than 3 channels found.")
                continue

        # Save the image as .png
        output_path = os.path.join(output_dir, f"{Path(tif_file).stem}.png")
        cv2.imwrite(output_path, img)
        print(f"Converted {tif_file} to {output_path}")

    print("All .tif images have been converted to .png.")


if __name__ == "__main__":
    # Input directory containing .tif images
    input_dir = "/home/nehoray/PycharmProjects/Segmentation_Models_Factory/data/images"

    # Output directory where .png images will be saved
    output_dir = "/home/nehoray/PycharmProjects/Segmentation_Models_Factory/data/images_png"

    # Call the conversion function
    convert_tif_to_png(input_dir, output_dir)
