import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from implementation.image_processing import ImageProcessing
from implementation.opencv_image_processing import OpenCVImageProcessing


class CatImage:

    #инкапсуляция изображения кота и его метаданных

    def __init__(self, image_data: np.ndarray, url: str, breed: str, image_id: str = ""):
        self._image_data = image_data
        self._url = url
        self._breed = breed
        self._id = image_id
        self._manual_processor = ImageProcessing()
        self._opencv_processor = OpenCVImageProcessing()

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

    def convert_to_grayscale_manual(self) -> np.ndarray:
        return self._manual_processor.rgb_to_grayscale(self._image_data)

    def convert_to_grayscale_opencv(self) -> np.ndarray:
        return self._opencv_processor.rgb_to_grayscale(self._image_data)

    def __add__(self, other: 'CatImage') -> 'CatImage':
        if self._image_data.shape != other._image_data.shape:
            raise ValueError("Images must have the same dimensions for addition")

        result = cv2.add(self._image_data, other._image_data)
        return CatImage(result, f"combined_{self._breed}", self._breed)

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        if self._image_data.shape != other._image_data.shape:
            raise ValueError("Images must have the same dimensions for subtraction")

        result = cv2.subtract(self._image_data, other._image_data)
        return CatImage(result, f"subtracted_{self._breed}", self._breed)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], url: str, breed: str, image_id: str = "") -> 'CatImage':
        image_data = cv2.imread(str(file_path))
        if image_data is None:
            raise ValueError(f"Could not load image from {file_path}")
        return cls(image_data, url, breed, image_id)