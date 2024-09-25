from abc import ABC, abstractmethod

class Filter(ABC):
    """
    Abstract class for an object that filters the noise from galaxy images.
    """
    @abstractmethod
    def filter(self, img):
        """This function will be responsible for filtering the noise from a given image.

        Args:
            img (ndarray): Image from which to filter the noise
        """
        pass

    @abstractmethod
    def additional_filter(self, img):
        """This method is used to conduct any additional filtering that might be necessary

        Args:
            img (ndarray): The image to filter.
        """
        pass