from ModelsFactory.base_model import BaseModel
from abc import abstractmethod
import numpy as np
import cv2


class SegmentationBaseModel(BaseModel):
    """
    Abstract base class for segmentation models.
    Inherits from BaseModel and adds the get_masks method.
    """

    def __init__(self, prompt: str = None):
        super().__init__(prompt)
        self.model = None
        self.image = None

    @abstractmethod
    def init_model(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def set_prompt(self, prompt: str):
        self.prompt = prompt

    @abstractmethod
    def set_image(self, image):
        """
        Set the input image for the model.
        Args:
            image: Input image to be processed by the model.
        """
        self.image = image

    @abstractmethod
    def get_result(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def get_masks(self) -> dict:
        """
        Retrieve the segmentation masks from the model's output.
        This should be implemented by all segmentation models.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def save_colored_result(self, output_path: str):
        """
        Save the colored segmentation mask result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    def save_result(self, output_path: str):
        """
        Save the segmentation mask result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        if self.image is None:
            raise ValueError("No image set. Please set an image before saving the result.")

        masks_data = self.get_masks()
        masks = masks_data.get("masks", [])
        labels = masks_data.get("labels", [])

        if not masks:
            raise ValueError("No masks available to save.")

        # Combine all masks into a single image for visualization
        combined_mask = np.zeros((self.image.shape[1], self.image.shape[2]),
                                 dtype=np.uint8)  # Use spatial dimensions of the image
        for idx, mask in enumerate(masks):
            if mask.shape != combined_mask.shape:
                mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            combined_mask[mask > 0] = (
                                              idx + 1) * 40  # Assign different intensity for each class to improve visualization

        result_image = cv2.merge([combined_mask, combined_mask, combined_mask])  # Convert to 3-channel image for saving
        cv2.imwrite(output_path, result_image)
        print(f"Segmentation mask saved to {output_path}")

    def format_segmentation_result(self, result: np.ndarray, class_mapping: dict) -> dict:
        """
        Format the segmentation result to ensure consistency between different segmentation models.
        This method ensures the result is formatted in the same way regardless of whether the segmentation is semantic or instance.

        Args:
            result (np.ndarray): The raw output of the segmentation model.
            class_mapping (dict): A dictionary mapping class indices to class labels.

        Returns:
            dict: A dictionary containing:
                - "masks" (List[np.ndarray]): List of binary masks for each class.
                - "labels" (List[str]): List of class labels corresponding to each mask.
        """
        # Convert raw model output to class predictions
        if result.ndim == 3:
            # Assuming the result is of shape [num_classes, height, width]
            predicted_classes = np.argmax(result, axis=0)
        else:
            predicted_classes = result

        masks = []
        labels = []
        unique_classes = np.unique(predicted_classes)
        for class_index in unique_classes:
            if class_index in class_mapping:
                mask = (predicted_classes == class_index).astype(np.uint8)
                masks.append(mask)
                labels.append(class_mapping[class_index])

        return {"masks": masks, "labels": labels}
