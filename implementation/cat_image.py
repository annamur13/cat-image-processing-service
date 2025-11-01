import cv2
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Dict, Any
from implementation.image_processing import ImageProcessing
from implementation.opencv_image_processing import OpenCVImageProcessing


class AbstractCatImage(ABC):

    def __init__(self, image_data: np.ndarray, url: str, breed: str, image_id: str = ""):
        self._validate_image_data(image_data)
        self._image_data = image_data
        self._url = url
        self._breed = breed
        self._id = image_id
        self._manual_processor = ImageProcessing()
        self._opencv_processor = OpenCVImageProcessing()

    @abstractmethod
    def _validate_image_data(self, image_data: np.ndarray) -> None:
        pass

    @property
    def image_data(self) -> np.ndarray:
        return self._image_data

    @property
    def url(self) -> str:
        return self._url

    @property
    def breed(self) -> str:
        return self._breed

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def detect_edges_manual(self, threshold: float = 0.2) -> np.ndarray:
        pass

    @abstractmethod
    def detect_edges_opencv(self) -> np.ndarray:
        pass

    @abstractmethod
    def apply_convolution_manual(self, kernel_type: str = "blur") -> np.ndarray:
        pass

    @abstractmethod
    def apply_convolution_opencv(self, kernel_type: str = "blur") -> np.ndarray:
        pass

    @abstractmethod
    def convert_to_grayscale_manual(self) -> np.ndarray:
        pass

    @abstractmethod
    def convert_to_grayscale_opencv(self) -> np.ndarray:
        pass

    def __add__(self, other: 'AbstractCatImage') -> 'AbstractCatImage':
        if self._image_data.shape != other._image_data.shape:
            raise ValueError("Images must have the same dimensions for addition")

        result = cv2.add(self._image_data, other._image_data)
        return self._create_combined_image(result, f"combined_{self._breed}")

    def __sub__(self, other: 'AbstractCatImage') -> 'AbstractCatImage':
        if self._image_data.shape != other._image_data.shape:
            raise ValueError("Images must have the same dimensions for subtraction")

        result = cv2.subtract(self._image_data, other._image_data)
        return self._create_combined_image(result, f"subtracted_{self._breed}")

    @abstractmethod
    def _create_combined_image(self, image_data: np.ndarray, operation_prefix: str) -> 'AbstractCatImage':
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_path: Union[str, Path], url: str, breed: str, image_id: str = "") -> 'AbstractCatImage':
        pass


class ColorCatImage(AbstractCatImage):

    def _validate_image_data(self, image_data: np.ndarray) -> None:
        if len(image_data.shape) != 3 or image_data.shape[2] != 3:
            raise ValueError("Color image must have 3 channels (RGB/BGR)")

    def detect_edges_manual(self, threshold: float = 0.2) -> np.ndarray:
        return self._manual_processor.edge_detection(self._image_data, threshold)

    def detect_edges_opencv(self) -> np.ndarray:
        gray = self._opencv_processor.rgb_to_grayscale(self._image_data)
        return self._opencv_processor.sobel_edge_detection(gray)

    def apply_convolution_manual(self, kernel_type: str = "blur") -> np.ndarray:
        return self._manual_processor.convolution(self._image_data, kernel_type)

    def apply_convolution_opencv(self, kernel_type: str = "blur") -> np.ndarray:
        if kernel_type == "blur":
            return self._opencv_processor.gaussian_blur(self._image_data)
        elif kernel_type == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return self._opencv_processor.convolution(self._image_data, kernel)
        elif kernel_type == "sobel":
            return self._opencv_processor.sobel_edge_detection(self._image_data)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def convert_to_grayscale_manual(self) -> np.ndarray:
        return self._manual_processor.rgb_to_grayscale(self._image_data)

    def convert_to_grayscale_opencv(self) -> np.ndarray:
        return self._opencv_processor.rgb_to_grayscale(self._image_data)

    def _create_combined_image(self, image_data: np.ndarray, operation_prefix: str) -> 'ColorCatImage':
        return ColorCatImage(image_data, f"{operation_prefix}_{self._breed}", self._breed)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], url: str, breed: str, image_id: str = "") -> 'ColorCatImage':
        image_data = cv2.imread(str(file_path))
        if image_data is None:
            raise ValueError(f"Could not load image from {file_path}")
        return cls(image_data, url, breed, image_id)


class GrayscaleCatImage(AbstractCatImage):
    def _validate_image_data(self, image_data: np.ndarray) -> None:
        if len(image_data.shape) != 2:
            raise ValueError("Grayscale image must have 2 dimensions")

    def detect_edges_manual(self, threshold: float = 0.2) -> np.ndarray:
        return self._manual_processor.edge_detection(self._image_data, threshold)

    def detect_edges_opencv(self) -> np.ndarray:
        return self._opencv_processor.sobel_edge_detection(self._image_data)

    def apply_convolution_manual(self, kernel_type: str = "blur") -> np.ndarray:
        return self._manual_processor.convolution(self._image_data, kernel_type)

    def apply_convolution_opencv(self, kernel_type: str = "blur") -> np.ndarray:
        if kernel_type == "blur":
            return self._opencv_processor.gaussian_blur(self._image_data)
        elif kernel_type == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return self._opencv_processor.convolution(self._image_data, kernel)
        elif kernel_type == "sobel":
            return self._opencv_processor.sobel_edge_detection(self._image_data)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def convert_to_grayscale_manual(self) -> np.ndarray:
        return self._image_data.copy()

    def convert_to_grayscale_opencv(self) -> np.ndarray:
        return self._image_data.copy()

    def _create_combined_image(self, image_data: np.ndarray, operation_prefix: str) -> 'GrayscaleCatImage':
        return GrayscaleCatImage(image_data, f"{operation_prefix}_{self._breed}", self._breed)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], url: str, breed: str, image_id: str = "") -> 'GrayscaleCatImage':
        image_data = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image_data is None:
            raise ValueError(f"Could not load image from {file_path}")
        return cls(image_data, url, breed, image_id)

    @classmethod
    def from_color_image(cls, color_image: ColorCatImage) -> 'GrayscaleCatImage':
        grayscale_data = color_image.convert_to_grayscale_opencv()
        return cls(grayscale_data, color_image.url, color_image.breed, color_image.id)