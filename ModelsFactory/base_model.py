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
        pass

    @abstractmethod
    def set_prompt(self, prompt: str):
        pass

    @abstractmethod
    def set_image(self, image):
        pass

    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def save_result(self, output_path: str):
        """
        Save the inference result to the specified output path.

        Args:
            output_path (str): The path to save the result image.
        """
        pass

    @staticmethod
    def get_available_classes() -> Union[list, str]:
        """
        Return the available classes for the model.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")



