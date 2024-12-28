import argparse
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import groundingdino.datasets.transforms as T
from common.model_name_registry import ModelNameRegistryDetection, PROMPT_MODEL
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from ModelsFactory.Detection.detection_base_model import DetectionBaseModel

class GroundingDINO_Model(DetectionBaseModel):
    """
    GroundingDINO_Model handles inference using the GroundingDINO API, inheriting from DetectionBaseModel.
    """
    def __init__(self, model_config_path: str, model_checkpoint_path: str, cpu_only: bool = False, model_name=ModelNameRegistryDetection.DINO.value):
        super().__init__(prompt=None)
        self.model_config_path = model_config_path
        self.model_checkpoint_path = model_checkpoint_path
        self.cpu_only = cpu_only
        self.model = None
        self.image = None
        self.image_pil = None
        self.inference_results = None
        # If the model_name is set, change the self.name to the desired one, like for example - opengeos
        self.model_name = model_name  # Assign model name

    def init_model(self):
        """
        Initialize the GroundingDINO model.
        """
        args = SLConfig.fromfile(self.model_config_path)
        args.device = "cuda" if not self.cpu_only else "cpu"
        self.model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        self.model.eval()
        if not self.cpu_only:
            self.model = self.model.to("cuda")
        print(f"Initialized {self.model_name} model with config: {self.model_config_path} and checkpoint: {self.model_checkpoint_path}")

    def set_prompt(self, prompt: List[str]):
        """
        Set the text prompt for object detection.

        Args:
            prompt (str): The text prompt for object detection.
        """
        # From BaseModel
        self.validate_prompt(prompt)
        prompt = ". ".join(prompt) + "."
        self.prompt = prompt

    def set_image(self, image: np.ndarray):
        """
        Set the input image for the GroundingDINO model to process.

        Args:
            image (np.ndarray): The image as a NumPy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Expected image as a NumPy array.")

        # Convert from NumPy (OpenCV format) to PIL (RGB)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply transformations
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)  # Convert to tensor: [3, h, w]

        self.image = image_tensor
        self.image_pil = image_pil

    def get_result(self) -> dict:
        """
        Run inference on the image and return the inference result.

        Returns:
            dict: The inference results from the GroundingDINO model.
        """
        if self.image is None or self.prompt is None:
            raise ValueError("Image or prompt not set. Please set both before calling get_result.")

        caption = self.prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        device = "cuda" if not self.cpu_only else "cpu"
        self.model = self.model.to(device)
        self.image = self.image.to(device)

        with torch.no_grad():
            outputs = self.model(self.image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # Filter output
        box_threshold = 0.3
        text_threshold = 0.25
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # Get phrase
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)

        # Store filtered logits (confidence scores) as part of inference_results
        scores = logits_filt.max(dim=1)[0].tolist()

        self.inference_results = {
            "boxes": boxes_filt,
            "labels": pred_phrases,
            "scores": scores,
            "size": [self.image_pil.height, self.image_pil.width],  # H, W
        }
        return self.inference_results

    def get_boxes(self) -> dict:
        """
        Extract bounding boxes, labels, and confidence scores from the inference result.

        Returns:
            dict: A dictionary containing:
                - "bboxes" (List[List[float]]): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
                - "labels" (List[str]): List of class labels corresponding to each bounding box.
                - "scores" (List[float]): List of confidence scores corresponding to each bounding box.
        """
        if self.inference_results is None:
            raise ValueError("No inference results found. Please run get_result() first.")

        formatted_result = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        H, W = self.inference_results["size"]
        for box, label, score in zip(self.inference_results["boxes"], self.inference_results["labels"], self.inference_results["scores"]):
            # Convert box from [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            formatted_result["bboxes"].append([x0, y0, x1, y1])
            formatted_result["labels"].append(label)
            formatted_result["scores"].append(score)  # Properly return the confidence score between 0 and 1

        return formatted_result

    def save_result(self, output_path: str):
        """
        Save the visualization of bounding boxes and labels to the specified output path.

        Args:
            output_path (str): The path to save the visualization image.
        """
        if self.inference_results is None:
            raise ValueError("No inference results found. Please run get_result() first.")

        draw = ImageDraw.Draw(self.image_pil)
        for box, label in zip(self.inference_results["boxes"], self.inference_results["labels"]):
            H, W = self.inference_results["size"]
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0, y0), label, fill=color)

        self.image_pil.save(output_path)
        print(f"Result saved to {output_path}")

    @staticmethod
    def get_available_classes() -> str:
        """
        Return a notice that this model uses a free prompt for object detection.

        Returns:
        - str: Notice string.
        """
        return PROMPT_MODEL

# Simple usage example of GroundingDINO_Model
if __name__ == "__main__":
    model_config_path = "/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/GroundingDINO_workspace/git_workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_checkpoint_path = "/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/GroundingDINO_workspace/git_workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"
    output_path = "/home/nehoray/PycharmProjects/UniversaLabeler/ModelsFactory/Detection/GroundingDINO_workspace/git_workspace/GroundingDINO/results/image.png"

    model = GroundingDINO_Model(model_config_path, model_checkpoint_path)
    model.init_model()
    model.set_prompt(["car", "bus", "truck", "bike"])
    model.set_image(cv2.imread(image_path))
    model.get_result()
    model.get_boxes()
    model.save_result(output_path)
