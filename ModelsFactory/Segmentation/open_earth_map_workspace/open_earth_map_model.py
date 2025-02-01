import cv2
import torch
import numpy as np
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from ModelsFactory.Segmentation.open_earth_map_workspace.git_workspace.open_earth_map import open_earth_map as oem
from common.model_name_registry import ConfigParameters, ModelNameRegistrySegmentation


class OpenEarthMapModel(SegmentationBaseModel):
    """
    OpenEarthMapModel is a segmentation model that inherits from SegmentationBaseModel.
    It uses a custom network architecture and the provided utils for inference.
    """

    def __init__(self, model_dir, model_checkpoint_name, in_channels=3, n_classes=6):
        super().__init__()
        self.model = None  # Model will be initialized in init_model()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.image = None
        self.result = None  # Cache for model output
        self.masks = None  # Cache for processed masks
        self.labels = None  # Cache for processed labels
        self.model_dir = model_dir
        self.model_checkpoint_name = model_checkpoint_name
        self.model_name = ModelNameRegistrySegmentation.OPEN_EARTH_MAP.value  # Assign model name

    CLASS_MAPPING = {
        0: "unknown",
        1: "greenery",  # includes Grass, Tree, Bareland, Cropland
        2: "pavement",
        3: "road",
        4: "buildings",
        5: "water",
    }

    @classmethod
    def get_available_classes(cls) -> list:
        return list(cls.CLASS_MAPPING.values())

    def set_advanced_parameters(self):
        print(f"{self.__class__.__name__} does not have advanced parameters.")

    def init_model(self):
        """
        Initialize the OpenEarthMap model.
        """
        self.model = oem.networks.UNetFormer(in_channels=self.in_channels, n_classes=self.n_classes)
        self.model = oem.utils.load_checkpoint(self.model, model_checkpoint_name=self.model_checkpoint_name,
                                               model_dir=self.model_dir)
        self.model.eval()
        print("OpenEarthMap model initialized.")

    def set_prompt(self, prompt: str):
        """
        Set any metadata or prompt information (not used in this specific case).
        """
        self.prompt = prompt

    def set_image(self, image: np.ndarray):
        """
        Set the input image that will be segmented by the model.

        Parameters:
        - image (np.ndarray): Input image to be processed.
        """
        # Resize all images to a fixed dimension, e.g., 1024x1024
        desired_size = (1024, 1024)
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)

        # Preprocess the image: Convert to RGB, normalize, and convert to tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

        # Add padding if necessary to ensure divisible by 32
        if image_tensor.shape[2] % 32 != 0 or image_tensor.shape[3] % 32 != 0:
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            new_h, new_w = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
            pad_h, pad_w = new_h - h, new_w - w
            image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        self.image = image_tensor
        self.result = None  # Reset cached result when a new image is set
        self.masks = None  # Reset cached masks
        self.labels = None  # Reset cached labels

    def get_result(self):
        """
        Get the segmentation result after processing the image with the model.

        Returns:
        - torch.Tensor: The raw output of the model (e.g., class probabilities or logits).
        """
        if self.result is not None:
            return self.result

        if self.model is None or self.image is None:
            raise ValueError("Model or image not set. Please initialize the model and set an image before getting the result.")

        with torch.no_grad():
            self.result = self.model(self.image.to(next(self.model.parameters()).device)).squeeze(0).cpu()
        return self.result

    def get_masks(self) -> dict:
        """
        Get the segmentation masks from the model's output.

        Returns:
        - dict: A dictionary containing:
            - "masks" (List[np.ndarray]): List of binary masks for each class.
            - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        if self.masks is not None and self.labels is not None:
            return {"masks": self.masks, "labels": self.labels}

        result = self.get_result()
        predicted_classes = np.argmax(result.numpy(), axis=0)
        formatted_result = self.format_segmentation_result(predicted_classes, self.CLASS_MAPPING)
        self.masks, self.labels = formatted_result["masks"], formatted_result["labels"]
        return formatted_result

    def save_colored_result(self, output_path: str):
        """
        Save the colored segmentation mask result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        if self.image is None:
            raise ValueError("No image set. Please set an image before saving the result.")

        result = self.get_result()
        predicted_classes = np.argmax(result.numpy(), axis=0)

        # Create an empty RGB image to visualize segmentation
        colored_result = np.zeros((predicted_classes.shape[0], predicted_classes.shape[1], 3), dtype=np.uint8)

        # Assign RGB values based on the predicted classes
        class_rgb_oem = {
            "unknown": [0, 0, 0],  # black
            "greenery": [0, 255, 36],  # bright green (merged category)
            "pavement": [148, 148, 148],  # gray
            "road": [255, 255, 255],  # white
            "buildings": [222, 31, 7],  # red
            "water": [0, 0, 255],  # blue
        }

        for class_name, color in class_rgb_oem.items():
            class_index = {v: k for k, v in self.CLASS_MAPPING.items()}[class_name]
            colored_result[predicted_classes == class_index] = color

        # Save the resulting RGB mask image
        cv2.imwrite(output_path, colored_result)
        print(f"Colored segmentation mask saved to {output_path}")


if __name__ == '__main__':
    model = OpenEarthMapModel(model_dir=ConfigParameters.OPEN_EARTH_MAP_MODEL_DIR.value,
                              model_checkpoint_name=ConfigParameters.OPEN_EARTH_MAP_MODEL_NAME.value)
    model.init_model()
    model.set_image(cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/images/mix/small_car.jpeg"))
    model.get_result()
    a = model.get_masks()
    b = model.get_bbox_from_masks()
    model.save_colored_result("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Segmentation/open_earth_map_workspace/test.png")