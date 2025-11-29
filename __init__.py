"""
Cat Image Processor Package
A package for downloading and processing cat images from TheCatAPI.
"""

__version__ = "1.0.0"
__author__ = "Murzina Anna"

from implementation.cat_image import CatImage
from implementation.cat_image_processor import CatImageProcessor
from implementation.image_processing import ImageProcessing
from implementation.opencv_image_processing import OpenCVImageProcessing

from logging_config import logger

__all__ = [
    'CatImage',
    'CatImageProcessor',
    'ImageProcessing',
    'OpenCVImageProcessing',
    'logger'
]