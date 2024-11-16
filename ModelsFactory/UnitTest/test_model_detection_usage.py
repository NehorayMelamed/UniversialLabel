import cv2

from ModelsFactory.Detection.Alfred_detection_workspace.alfred_detection_nodel import AlfredDetectionModel
from ModelsFactory.Detection.Waldo_workspace.waldo_model_detection import WaldoDetectionModel
from ModelsFactory.Detection.YOLO_WORLD_workspace.yolo_world_model import YOLOWorld_Model


## yolo - world
def simple_usage_example_yolo_world():
    # Step 1: Instantiate the YOLOWorld_Model
    model_path = "/home/nehoray/PycharmProjects/UniversaLabeler/common/weights/yolov8s-world.pt"  # Replace with the actual path to the model weights
    yolo_world = YOLOWorld_Model(model_path=model_path, verbose=True)

    # Step 2: Initialize the model
    yolo_world.init_model()

    # Step 3: Set a prompt (classes of interest)
    custom_classes = ["person", "car", "tree", "Building", "Road", "Vegetation"]
    yolo_world.set_prompt(custom_classes)

    # Step 4: Load an image from file
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"  # Replace with the actual path to your image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Please check the path.")
    yolo_world.set_image(image)

    # Step 5: Run inference and get results
    results = yolo_world.get_result()
    print("Inference Results:", results)

    # Step 6: Extract bounding boxes, labels, and scores
    boxes = yolo_world.get_boxes()
    print("Bounding Boxes:", boxes)

    yolo_world.save_result("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/UnitTest/results/detection/yolo_world/img1.png")


# # yolo alfred
# def simple_usage_example_yolo_alfred():
#     # Step 1: Instantiate the YOLOWorld_Model
#     alfred = AlfredDetectionModel()
#
#     # Step 2: Initialize the model
#     alfred.init_model()
#
#     # Step 3: Set a prompt (classes of interest)
#     # custom_classes = ["person", "bus", "bicycle", "car"]
#     # alfred.set_prompt(custom_classes)
#
#     # Step 4: Load an image from file
#     image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/images/mix/small_car.jpeg"  # Replace with the actual path to your image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at path {image_path} could not be loaded. Please check the path.")
#     alfred.set_image(image)
#
#     # Step 5: Run inference and get results
#     results = alfred.get_result()
#     print("Inference Results:", results)
#
#     # Step 6: Extract bounding boxes, labels, and scores
#     boxes = alfred.get_boxes()
#     print("Bounding Boxes:", boxes)
#
#     alfred.save_result("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/UnitTest/results/detection/alfreds/img1.png")
#
#
# # yolo - waldo
# def simple_usage_example_yolo_waldo():
#     # Step 1: Instantiate the YOLOWorld_Model
#     waldo = WaldoDetectionModel()
#
#     # Step 2: Initialize the model
#     waldo.init_model()
#
#     # Step 3: Set a prompt (classes of interest)
#     # custom_classes = ["person", "bus", "bicycle", "car"]
#     # alfred.set_prompt(custom_classes)
#
#     # Step 4: Load an image from file
#     image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/images/mix/small_car.jpeg"  # Replace with the actual path to your image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at path {image_path} could not be loaded. Please check the path.")
#     waldo.set_image(image)
#
#     # Step 5: Run inference and get results
#     results = waldo.get_result()
#     print("Inference Results:", results)
#
#     # Step 6: Extract bounding boxes, labels, and scores
#     boxes = waldo.get_boxes()
#     print("Bounding Boxes:", boxes)
#
#     waldo.save_result("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/UnitTest/results/detection/waldo/img1.png")


if __name__ == '__main__':
    simple_usage_example_yolo_alfred()
    # simple_usage_example_yolo_world()
    # simple_usage_example_yolo_waldo()