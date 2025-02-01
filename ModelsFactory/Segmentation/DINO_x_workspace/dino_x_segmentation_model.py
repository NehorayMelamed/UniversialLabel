import os
from typing import List
import cv2
import numpy as np
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from common.model_name_registry import ModelNameRegistrySegmentation, PROMPT_MODEL


class DINOXSegmentation(SegmentationBaseModel):
    def __init__(self, api_token: str):
        super().__init__()
        self.api_token = api_token
        self.client = None
        self.config = None
        self.model_name = ModelNameRegistrySegmentation.DINOX_SEGMENTATION.value
        self.prompts = None
        self.image_url = None
        self.inference_results = None
        self.original_image = None  # Store the original image

    def init_model(self):
        """
        Initialize the DINO-X API client.
        """
        self.config = Config(self.api_token)
        self.client = Client(self.config)

    def set_advanced_parameters(self):
        print(f"{self.__class__.__name__} does not have advanced parameters.")

    def set_prompt(self, prompt: List[str]):
        """
        Set the text prompts for DINO-X.

        Args:
            prompt (List[str]): A list of custom class names for object detection.
        """
        self.validate_prompt(prompt)
        prompts_str = ". ".join(prompt) + "."
        self.prompts = [TextPrompt(text=prompts_str)]

    def set_image(self, image: np.ndarray):
        """
        Upload the input image to the DDS Cloud API using in-memory data and set the image URL.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        self.original_image = image.copy()  # Store the original image
        _, buffer = cv2.imencode('.png', image)
        local_file = "./temp_image.png"
        with open(local_file, 'wb') as f:
            f.write(buffer.tobytes())
        self.image_url = self.client.upload_file(local_file)
        os.remove(local_file)

        if not self.image_url:
            raise ValueError("Failed to upload image to DDS Cloud API.")

    def get_result(self):
        """
        Run inference using DINO-X and return raw results.

        Returns:
            list: Segmentation objects with bounding boxes and masks.
        """
        if self.image_url is None or self.prompts is None:
            raise ValueError("Image URL or prompts are not set.")

        task = DinoxTask(
            image_url=self.image_url,
            prompts=self.prompts,
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox, DetectionTarget.Mask]
        )
        self.client.run_task(task)
        self.inference_results = task.result.objects
        return self.inference_results

    def get_masks(self) -> dict:
        """
        Extract segmentation masks from the inference results.

        Returns:
            dict: Contains:
                - "masks": List of binary masks (np.ndarray).
                - "labels": List of corresponding class labels.
        """
        if not self.inference_results:
            raise ValueError("No inference results found. Please run get_result first.")

        masks = []
        labels = []

        for obj in self.inference_results:
            mask = DetectionTask.rle2mask(DetectionTask.string2rle(obj.mask.counts), obj.mask.size)
            masks.append(mask)
            labels.append(obj.category)

        return {"masks": masks, "labels": labels}

    def save_colored_result(self, output_path: str):
        """
        Save the segmentation results as a colored image.

        Args:
            output_path (str): Path to save the annotated image.
        """
        if not self.inference_results:
            raise ValueError("No inference results found. Please run get_result first.")
        if self.original_image is None:
            raise ValueError("Original image is not available.")

        img = self.original_image.copy()  # Use the original image
        masks, labels = [], []
        for obj in self.inference_results:
            masks.append(DetectionTask.rle2mask(DetectionTask.string2rle(obj.mask.counts), obj.mask.size))
            labels.append(obj.category)

        for mask, label in zip(masks, labels):
            color = np.random.randint(0, 255, size=(3,), dtype=int)
            img[mask > 0] = color

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that this model uses a free prompt for object detection.

        Returns:
        - str: Notice string.
        """
        return PROMPT_MODEL


if __name__ == '__main__':
    dinox_segmentation = DINOXSegmentation(api_token="3fabc3dcc385e7deb0067833ff9da337")
    dinox_segmentation.init_model()
    dinox_segmentation.set_prompt(["wheel", "eye", "helmet", "mouse", "mouth", "vehicle", "steering", "ear", "nose"])
    dinox_segmentation.set_image(cv2.imread("/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/DINO_X_workspace/DINO-X-API/assets/demo.png"))
    results = dinox_segmentation.get_result()

    # Print masks and labels
    masks = dinox_segmentation.get_masks()
    print(masks)

    # Save annotated result
    dinox_segmentation.save_colored_result("out/segmented_image.jpg")
