from abc import ABC, abstractmethod


class UniversalLabeler(ABC):
    """
    UniversalLabelerInterface defines the methods that any labeler (detection or segmentation) must implement.
    """

    @abstractmethod
    def process_image(self, image):
        """
        Process the input image using the respective models and return the combined result.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def _create_models(self):
        """
        Create the required models (either detection or segmentation) using the factory interface.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

    @abstractmethod
    def _combine_results(self, results):
        """
        Combine the results from multiple models (e.g., NMS for detection, mask combination for segmentation).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support functionality.")

