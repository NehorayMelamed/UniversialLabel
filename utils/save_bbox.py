import cv2
import numpy as np


def select_bbox(image_path: str) -> np.ndarray:
    """
    Open an image and allow the user to select a bounding box with the mouse.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Bounding box in [x_min, y_min, x_max, y_max] format.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    # Display instructions
    print("Select a ROI (bounding box) and press ENTER or SPACE.")
    print("To cancel the selection, press C.")

    # Open a window and let the user select the ROI
    bbox = cv2.selectROI("Select BBOX", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Check if a valid ROI was selected
    if bbox[2] == 0 or bbox[3] == 0:
        print("No valid BBOX selected. Exiting.")
        return None

    # Convert the bbox to [x_min, y_min, x_max, y_max] format
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    return np.array([x_min, y_min, x_max, y_max])


if __name__ == "__main__":
    # Path to the input image
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"

    try:
        bbox = select_bbox(image_path)
        if bbox is not None:
            print(f"Selected BBOX: {bbox}")
    except Exception as e:
        print(f"Error: {e}")
