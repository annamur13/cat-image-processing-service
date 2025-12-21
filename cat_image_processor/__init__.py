# cat_image_processor/__init__.py

from cat_image_processor.implementation.image_processing import ImageProcessing
from cat_image_processor.implementation.opencv_image_processing import OpenCVImageProcessing
from cat_image_processor.implementation import CatImageProcessor
from cat_image_processor.implementation import CatImage
from cat_image_processor.logging_config import logger
from . import implementation

__version__ = "1.0.0"
__all__ = [
    "ImageProcessing",
    "OpenCVImageProcessing",
    "CatImageProcessor",
    "CatImage",
    "logger"
]