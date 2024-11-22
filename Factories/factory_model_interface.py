from abc import ABC, abstractmethod
from typing import Dict
from ModelsFactory.base_model import BaseModel


class FactoryModelInterface(ABC):
    """
    Abstract Factory Model Interface that defines the create_model method.
    This method will be implemented by segmentation and detection factories.
    """
    def __init__(self, model_mapping: Dict[str, BaseModel]):
        """
        Initialize the factory base with a model mapping.

        Parameters:
        - model_mapping (dict): A dictionary where keys are model names and values are model classes.
        """
        self.model_mapping: Dict[str, BaseModel] = model_mapping

    @abstractmethod
    def create_model(self, model_type: str):
        """
        Create and return a model instance based on the model_type.

        Parameters:
        - model_type (str): The type of model to create.

        Returns:
        - An instance of the model (either segmentation or detection based).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    def available_models(self) -> list:
        """
        Return a list of available models.

        Returns:
        - list: List of model names available in the factory.
        """
        return list(self.model_mapping.keys())


    def get_available_models_with_classes(self) -> dict:
        """
        Get all available models along with their classes or prompt notice.

        Returns:
        - dict: A dictionary where keys are model names and values are lists of class names or "free prompt".
        """
        model_classes = {}
        for model_name, model_cls in self.model_mapping.items():
            # Call the static or class method to get the available classes
            available_classes = model_cls.get_available_classes()
            model_classes[model_name] = available_classes
        return model_classes