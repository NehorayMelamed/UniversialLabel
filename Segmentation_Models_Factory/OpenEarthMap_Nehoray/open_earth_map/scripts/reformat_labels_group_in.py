import cv2
import numpy as np
import os


class_grey_oem = {
    "unknown": 0,
    "bareland": 1,
    "grass": 2,
    "pavement": 3,
    "road": 4,
    "tree": 5,
    "water": 6,
    "cropland": 7,
    "buildings": 8,
}



# #### Define the new merged class mapping
new_class_mapping = {
    "unknown": 0,
    "greenery": 1,  # includes Grass, Tree, Bareland, Cropland
    "pavement": 2,
    "road": 3,
    "buildings": 4,
    "water": 5,
}


# Function to preprocess and convert the mask
def preprocess_mask(input_path, output_path):
    # Read the image (mask)
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Initialize a new mask with the same shape
    new_mask = np.zeros_like(mask)

    # Map the original classes to the new classes
    for class_name, class_value in class_grey_oem.items():
        if class_name in ["bareland", "grass", "tree", "cropland"]:
            new_mask[mask == class_value] = new_class_mapping["greenery"]
        elif class_name in new_class_mapping:
            new_mask[mask == class_value] = new_class_mapping[class_name]

    # Save the new mask to the output directory
    cv2.imwrite(output_path, new_mask)

# Function to loop through all files in a directory
def process_all_masks(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif"):  # Assuming mask files are in .tif format
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Process each mask
            preprocess_mask(input_path, output_path)
            print(f"Processed: {filename}")

# Example usage
input_dir = "/raid/open_earth_map/xdb_backup/labels"
output_dir = "/raid/open_earth_map/xbd/processed_grouped_labels"

process_all_masks(input_dir, output_dir)


