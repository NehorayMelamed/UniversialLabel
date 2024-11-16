import cv2
import os
import numpy as np


def simulate_thermal_image(raw_image_path, output_directory):
    # Step 1: Load the raw image (assuming it's in grayscale)
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

    if raw_image is None:
        raise ValueError("The image could not be loaded. Please check the file path.")

    # Step 2: Normalize the image to a range of 0 to 255
    normalized_image = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)

    # Step 3: Apply a thermal colormap
    thermal_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

    # Step 4: Extract the base name of the raw image (without extension) and build the output path
    base_name = os.path.basename(raw_image_path)
    base_name_without_ext = os.path.splitext(base_name)[0]
    output_image_path = os.path.join(output_directory, f"{base_name_without_ext}_thermal.jpg")

    # Step 5: Save the thermal image in the specified output directory
    cv2.imwrite(output_image_path, thermal_image)

    # Optionally, display the image (for debugging or visualization)
    # cv2.imshow('Thermal Image', thermal_image)
    # cv2.waitKey(0)  # Wait for a key press
    # cv2.destroyAllWindows()

    print(f"Thermal image saved at: {output_image_path}")

if __name__ == '__main__':
    # Example usage:
    raw_image_path = '/home/nehoray/PycharmProjects/Segmentation_Models_Factory/data/images/chiangmai_4.tif'  # Replace with your raw image path
    output_directory = 'output_simulated_thermal'  # Replace with your desired output directory
    simulate_thermal_image(raw_image_path, output_directory)
