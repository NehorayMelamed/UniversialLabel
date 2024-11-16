from abc import ABC, abstractmethod


class FactoryModelInterface(ABC):
    """
    Abstract Factory Model Interface that defines the create_model method.
    This method will be implemented by segmentation and detection factories.
    """

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

    @abstractmethod
    def available_models(self):
        """
        Return a list of available detection model types.

        Returns:
        - list: A list of model names available in the ModelNameRegistry.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")



