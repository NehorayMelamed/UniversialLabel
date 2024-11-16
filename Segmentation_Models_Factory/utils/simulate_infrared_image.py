import cv2
import os
import numpy as np


def simulate_infrared_image(raw_image_path, output_directory):
    # Step 1: Load the raw image (assuming it's in grayscale)
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

    if raw_image is None:
        raise ValueError("The image could not be loaded. Please check the file path.")

    # Step 2: Normalize the image to enhance temperature-like intensity (higher contrast)
    normalized_image = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)

    # Step 3: Optionally, apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)

    # Step 4: Apply a custom colormap for IR simulation
    # Using COLORMAP_BONE or COLORMAP_PINK for a subtle, grayscale-like IR effect
    infrared_image = cv2.applyColorMap(enhanced_image, cv2.COLORMAP_BONE)

    # Step 5: Extract the base name of the raw image (without extension) and build the output path
    base_name = os.path.basename(raw_image_path)
    base_name_without_ext = os.path.splitext(base_name)[0]
    output_image_path = os.path.join(output_directory, f"{base_name_without_ext}_infrared.jpg")

    # Step 6: Save the infrared image in the specified output directory
    cv2.imwrite(output_image_path, infrared_image)

    # Optionally, display the image (for debugging or visualization)
    # cv2.imshow('Infrared Image', infrared_image)
    # cv2.waitKey(0)  # Wait for a key press
    # cv2.destroyAllWindows()

    print(f"Infrared image saved at: {output_image_path}")

if __name__ == '__main__':
    raw_image_path = '/home/nehoray/PycharmProjects/Segmentation_Models_Factory/data/images/chiangmai_4.tif'  # Replace with your raw image path
    output_directory = 'output_simulated_ir'  # Replace with your desired output directory

    simulate_infrared_image(raw_image_path, output_directory)
    # Example usage:
