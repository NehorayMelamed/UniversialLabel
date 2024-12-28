import os
import cv2
import numpy as np
from typing import Dict, List, Union
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from ModelsFactory.Segmentation.segmentation_base_model import SegmentationBaseModel
from common.model_name_registry import ModelNameRegistrySegmentation


class SAMSegmentation(SegmentationBaseModel):
    """
    SAMSegmentation class integrates Facebook's Segment Anything Model (SAM).
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_name = ModelNameRegistrySegmentation.SAM.value
        self.sam_model = None
        self.mask_generator = None
        self.image = None
        self.masks = []

    def init_model(self):
        """
        Initialize the SAM model and mask generator.
        """
        sam_type = "vit_h" if "vit_h" in self.checkpoint_path else "vit_b"
        self.sam_model = sam_model_registry[sam_type](checkpoint=self.checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)

    def set_prompt(self, prompt: str):
        """
        SAM does not support text prompts, but the method is provided for compatibility.
        """
        print("SAM does not support setting classes. Only segmentation masks are generated.")

    def set_image(self, image: np.ndarray):
        """
        Set the input image to be segmented by SAM.

        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")
        self.image = image

    def get_result(self) -> List[Dict]:
        """
        Generate segmentation masks for the provided image.

        Returns:
            list: List of segmentation masks with additional metadata.
        """
        if self.image is None:
            raise ValueError("No image set. Please set an image before getting results.")

        self.masks = self.mask_generator.generate(self.image)
        return self.masks

    def get_masks(self) -> Dict[str, Union[List[np.ndarray], List[str]]]:
        """
        Extract segmentation masks in a consistent format.

        Returns:
            dict: Contains:
                - "masks": List of binary masks (np.ndarray).
                - "labels": Empty list, as SAM does not provide class labels.
        """
        if not self.masks:
            raise ValueError("No segmentation results found. Please run get_result first.")

        binary_masks = [mask["segmentation"] for mask in self.masks]
        return {"masks": binary_masks, "labels": []}

    def save_colored_result(self, output_path: str):
        """
        Save the segmentation results as a colored overlay image.

        Args:
            output_path (str): Path to save the result image.
        """
        if self.image is None:
            raise ValueError("No image set. Please set an image before saving results.")
        if not self.masks:
            raise ValueError("No segmentation results found. Please run get_result first.")

        colored_image = self.image.copy()
        for mask in self.masks:
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            colored_image[mask["segmentation"]] = (
                colored_image[mask["segmentation"]] * 0.5 + color * 0.5
            ).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_image)
        print(f"Colored segmentation mask saved to {output_path}")

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that this model uses free-form segmentation masks.

        Returns:
            str: Notice string.
        """
        return "SAM does not support predefined classes. It generates segmentation masks without labels."


if __name__ == "__main__":
    # Example usage
    sam_segmentation = SAMSegmentation(
        checkpoint_path="/home/nehoray/PycharmProjects/UniversaLabeler/common/weights/sam_vit_h_4b8939.pth"
    )
    sam_segmentation.init_model()
    image = cv2.imread("/home/nehoray/PycharmProjects/test_opengeos/test_image.png")
    sam_segmentation.set_image(image)

    # Generate results
    results = sam_segmentation.get_result()
    print(f"Generated {len(results)} segmentation masks.")

    # Get masks
    masks = sam_segmentation.get_masks()
    print(f"Extracted {len(masks['masks'])} binary masks.")

    # Save results
    sam_segmentation.save_colored_result("out/sam_segmented_image.jpg")
