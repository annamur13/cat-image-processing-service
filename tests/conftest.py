import pytest
import numpy as np
import cv2

@pytest.fixture
def sample_image_rgb():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def sample_image_grayscale():
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)

@pytest.fixture
def sample_cat_image():
    from implementation.cat_image import CatImage
    cat_image = CatImage(image_id="test_001", image_url="https://cdn2.thecatapi.com/images/J2PmlIizw.jpg")
    cat_image.image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return cat_image