import os
from abc import abstractmethod
from typing import Dict, List

import cv2

from ModelsFactory.base_model import BaseModel


class ImageCaption(BaseModel):
    """
    ImageCaption extends BaseModel with a specific method to get detection boxes.
    """

    def __init__(self, prompt: str = None):
        super().__init__(prompt)
        self.model = None

    @abstractmethod
    def init_model(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def set_prompt(self, prompt: List[str]):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")


    @abstractmethod
    def set_image(self, image):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def get_result(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")


    @abstractmethod
    def save_result(self, output_path: str):
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")
