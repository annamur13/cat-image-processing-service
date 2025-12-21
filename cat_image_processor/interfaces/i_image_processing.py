from abc import ABC, abstractmethod

import numpy as np


class IImageProcessing(ABC):

    @abstractmethod
    def convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        pass

    @abstractmethod
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        pass

