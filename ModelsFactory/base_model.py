from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import cv2
import os


class BaseModel(ABC):
    def __init__(self, prompt: str = None):
        """
        Initialize the BaseModel with a prompt.

        Args:
            prompt (str): The prompt used for initializing the model.
        """
        self.prompt = prompt
        self.image = None
        self.model_name = None


    @abstractmethod
    def init_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def set_prompt(self, prompt: str):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def set_image(self, image):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_result(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def save_result(self, output_path: str):
        """
        Save the inference result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def get_available_classes() -> Union[list, str]:
        """
        Return the available classes for the model.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


    @abstractmethod
    def set_advanced_parameters(self, **kwargs):
        """
        set advanced parameters for the model
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


    def validate_prompt(self, prompts):
        """
        Validate that the prompts are a list of strings.

        Args:
            prompts (Any): The input prompts to validate.

        Raises:
            ValueError: If the prompts are not a list of strings.
        """
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Prompts must be a list of strings.")


