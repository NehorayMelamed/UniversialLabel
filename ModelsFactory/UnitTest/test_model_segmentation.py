import cv2
from ModelsFactory.Segmentation.open_earth_map_workspace.open_earth_map_model import OpenEarthMapModel


def simple_usage_example_open_earth_map():
    # Step 1: Instantiate the OpenEarthMapModel
    open_earth_map_model = OpenEarthMapModel(in_channels=3, n_classes=6)

    # Step 2: Initialize the model
    open_earth_map_model.init_model()

    # Step 3: Load an image from file
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/Segmentation_Models_Factory/OpenEarthMap_Nehoray/open_earth_map/predictions/chiangmai_4/chiangmai_4_input.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Please check the path.")

    # Step 4: Set the image for the model
    open_earth_map_model.set_image(image)

    # Step 5: Run inference and get results
    result = open_earth_map_model.get_result()
    print(result)
    print("Inference Result (Raw):", result.shape)

    # Step 6: Extract segmentation masks
    masks = open_earth_map_model.get_masks()
    print("Segmentation Masks:", masks)

    # Step 7: Save the segmentation mask result
    output_path = "/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/UnitTest/results/segmentation/open_earth_map/open_earth_map_mask.png"
    open_earth_map_model.save_result(output_path)

# Run the simple usage example
if __name__ == "__main__":
    simple_usage_example_open_earth_map()
