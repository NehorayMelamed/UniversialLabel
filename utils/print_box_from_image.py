
import cv2
import matplotlib.pyplot as plt


def select_bbox_with_matplotlib(image_path):
    """
    Display the image using matplotlib and manually input BBOX coordinates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Load and display the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check the file path.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Manually select BBOX coordinates")
    plt.show()

    # Manually input the BBOX coordinates
    print("Enter BBOX coordinates (x_min, y_min, x_max, y_max):")
    x_min = int(input("x_min: "))
    y_min = int(input("y_min: "))
    x_max = int(input("x_max: "))
    y_max = int(input("y_max: "))

    return [x_min, y_min, x_max, y_max]


if __name__ == "__main__":
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"
    bbox = select_bbox_with_matplotlib(image_path)
    print("Selected BBOX:", bbox)

