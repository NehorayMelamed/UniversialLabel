import cv2
import torch
import numpy as np
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from git_workspace.sam2.sam2.build_sam import build_sam2
from git_workspace.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from common.model_name_registry import ConfigParameters


class SAM2Model(SegmentationBaseModel):
    """
    SAM2Model is a segmentation model that inherits from SegmentationBaseModel.
    It provides image segmentation capabilities using the SAM 2 foundation model.
    """

    def __init__(self, checkpoint_path: str, config_file: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.config_file = config_file
        self.model = None  # Model will be initialized in init_model()
        self.predictor = None
        self.image = None

    CLASS_MAPPING = {

    }

    @classmethod
    def get_available_classes(cls) -> list:
        return list(cls.CLASS_MAPPING.values())

    def init_model(self):
        """
        Initialize the SAM2 model.
        """
        # Load the model checkpoint and configuration
        self.model = build_sam2(self.config_file, self.checkpoint_path)
        self.predictor = SAM2ImagePredictor(self.model)
        self.model.eval()
        print("SAM2 model initialized.")

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
        # Preprocess the image: Convert to RGB, normalize, and convert to tensor
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        self.image = image_tensor

    def get_result(self):
        """
        Get the segmentation result after processing the image with the model.

        Returns:
        - torch.Tensor: The raw output of the model (e.g., class probabilities or logits).
        """
        if self.model is None or self.image is None:
            raise ValueError(
                "Model or image not set. Please initialize the model and set an image before getting the result.")

        with torch.no_grad():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(self.image)
                masks, _, _ = self.predictor.predict([])  # Provide an empty list as prompts for generic segmentation
        return masks

    def get_masks(self) -> dict:
        """
        Get the segmentation masks from the model's output.

        Returns:
        - dict: A dictionary containing:
            - "masks" (List[np.ndarray]): List of binary masks for each class.
            - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        result = self.get_result()
        # Extract binary masks for each class from the raw output
        predicted_classes = np.argmax(result.numpy(), axis=0)
        return self.format_segmentation_result(predicted_classes, self.CLASS_MAPPING)

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

        class_rgb = {
            "unknown": [0, 0, 0],  # black

        }

        # Convert to RGB mask
        colored_mask = np.zeros((predicted_classes.shape[0], predicted_classes.shape[1], 3), dtype=np.uint8)
        for label, rgb in class_rgb.items():
            colored_mask[predicted_classes == self.CLASS_MAPPING[label]] = rgb

        # Convert tensor image back to numpy and blend it with the colored mask
        original_image_np = self.image.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        blended_result = (0.7 * original_image_np + 0.3 * colored_mask).astype(np.uint8)

        # Save to file
        cv2.imwrite(output_path, blended_result)
        print(f"Colored segmentation mask saved to {output_path}")


if __name__ == '__main__':
    # Example usage of SAM2Model
    checkpoint_path = ConfigParameters.SAM2_CHECKPOINT_PATH.value
    config_file = ConfigParameters.SAM2_CONFIG_FILE.value

    model = SAM2Model(checkpoint_path=checkpoint_path, config_file=config_file)
    model.init_model()
    model.set_image(cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/data/images/mix/small_car.jpeg"))
    model.get_result()
    model.get_masks()
    model.save_colored_result(
        "/home/nehoray/PycharmProjects/UniversaLabeler/Segmentation_Models_Factory/SAM2/predictions/test.png")
