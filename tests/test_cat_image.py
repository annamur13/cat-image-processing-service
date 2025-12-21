import unittest
import os
import cv2
import numpy as np
from cat_image_processor import CatImage


class TestCatImage(unittest.TestCase):

    def setUp(self):
        self.test_image_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.cat_image = CatImage(
            image_data=self.test_image_rgb,
            url="https://cdn2.thecatapi.com/images/O2aNhFGU-.jpg",
            breed="Siamese",
            image_id="test_001"
        )
        self.test_image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.cat_image2 = CatImage(
            image_data=self.test_image2,
            url="https://cdn2.thecatapi.com/images/RhBsBQg6y.jpg",
            breed="Persian",
            image_id="test_002"
        )

    def tearDown(self):
        if hasattr(self, 'output_path') and os.path.exists(self.output_path):
            os.remove(self.output_path)

    # --- URL ---
    def test_url_is_correct(self):
        self.assertEqual(self.cat_image.url, "https://cdn2.thecatapi.com/images/O2aNhFGU-.jpg  ")

    # --- Breed ---
    def test_breed_is_correct(self):
        self.assertEqual(self.cat_image.breed, "Siamese")

    # --- Image ID ---
    def test_image_id_is_correct(self):
        self.assertEqual(self.cat_image.image_id, "test_001")

    # --- Image data ---
    def test_image_data_matches_input(self):
        np.testing.assert_array_equal(self.cat_image.image_data, self.test_image_rgb)

    # --- Edge detection (manual) ---
    def test_detect_edges_manual_returns_correct_shape(self):
        edges = self.cat_image.detect_edges_manual(threshold=0.2)
        self.assertEqual(edges.shape, self.test_image_rgb.shape[:2])

    def test_detect_edges_manual_returns_uint8(self):
        edges = self.cat_image.detect_edges_manual(threshold=0.2)
        self.assertEqual(edges.dtype, np.uint8)

    # --- Edge detection (OpenCV) ---
    def test_detect_edges_opencv_returns_2d_array(self):
        edges = self.cat_image.detect_edges_opencv()
        self.assertEqual(len(edges.shape), 2)

    def test_detect_edges_opencv_returns_correct_shape(self):
        edges = self.cat_image.detect_edges_opencv()
        self.assertEqual(edges.shape, self.test_image_rgb.shape[:2])

    def test_detect_edges_opencv_returns_uint8(self):
        edges = self.cat_image.detect_edges_opencv()
        self.assertEqual(edges.dtype, np.uint8)

    def test_apply_convolution_manual_returns_correct_shape_and_dtype(self):
        for kernel_type in ["blur", "sharpen", "sobel_x"]:
            with self.subTest(kernel_type=kernel_type):
                result = self.cat_image.apply_convolution_manual(kernel_type)
                self.assertEqual(result.shape, self.test_image_rgb.shape[:2])
                self.assertEqual(result.dtype, np.uint8)

    def test_apply_convolution_opencv_returns_correct_shape_and_dtype(self):
        for kernel_type in ["blur", "sharpen", "sobel"]:
            with self.subTest(kernel_type=kernel_type):
                result = self.cat_image.apply_convolution_opencv(kernel_type)
                expected_shape = self.test_image_rgb.shape[:2] if kernel_type == "sobel" else self.test_image_rgb.shape
                self.assertEqual(result.shape, expected_shape)
                self.assertEqual(result.dtype, np.uint8)

    # --- Grayscale ---
    def test_convert_to_grayscale_manual_returns_2d(self):
        grayscale = self.cat_image.convert_to_grayscale_manual()
        self.assertEqual(len(grayscale.shape), 2)

    def test_convert_to_grayscale_manual_returns_uint8(self):
        grayscale = self.cat_image.convert_to_grayscale_manual()
        self.assertEqual(grayscale.dtype, np.uint8)

    def test_convert_to_grayscale_opencv_returns_2d(self):
        grayscale = self.cat_image.convert_to_grayscale_opencv()
        self.assertEqual(len(grayscale.shape), 2)

    def test_convert_to_grayscale_opencv_returns_uint8(self):
        grayscale = self.cat_image.convert_to_grayscale_opencv()
        self.assertEqual(grayscale.dtype, np.uint8)

    # --- Image addition ---
    def test_image_addition_returns_catimage_instance(self):
        result = self.cat_image + self.cat_image2
        self.assertIsInstance(result, CatImage)

    def test_image_addition_preserves_breed_from_left(self):
        result = self.cat_image + self.cat_image2
        self.assertEqual(result.breed, "Siamese")

    def test_image_addition_url_contains_combined(self):
        result = self.cat_image + self.cat_image2
        self.assertIn("combined", result.url)

    def test_image_addition_preserves_shape(self):
        result = self.cat_image + self.cat_image2
        self.assertEqual(result.image_data.shape, self.test_image_rgb.shape)

    # --- Image subtraction ---
    def test_image_subtraction_returns_catimage_instance(self):
        result = self.cat_image - self.cat_image2
        self.assertIsInstance(result, CatImage)

    def test_image_subtraction_preserves_breed_from_left(self):
        result = self.cat_image - self.cat_image2
        self.assertEqual(result.breed, "Siamese")

    def test_image_subtraction_url_contains_subtracted(self):
        result = self.cat_image - self.cat_image2
        self.assertIn("subtracted", result.url)

    def test_image_subtraction_preserves_shape(self):
        result = self.cat_image - self.cat_image2
        self.assertEqual(result.image_data.shape, self.test_image_rgb.shape)

    # --- Error on size mismatch ---
    def test_addition_with_different_sizes_raises_value_error(self):
        small_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cat_image_small = CatImage(small_image, "url", "Breed", "id")
        with self.assertRaises(ValueError):
            _ = self.cat_image + cat_image_small

    # --- from_file ---
    def test_from_file_sets_correct_url(self):
        self.output_path = "test_image.png"
        cv2.imwrite(self.output_path, self.test_image_rgb)
        cat_img = CatImage.from_file(self.output_path, "https://example.com/cat.jpg", "TestBreed", "file_id")
        self.assertEqual(cat_img.url, "https://example.com/cat.jpg")

    def test_from_file_sets_correct_breed(self):
        self.output_path = "test_image.png"
        cv2.imwrite(self.output_path, self.test_image_rgb)
        cat_img = CatImage.from_file(self.output_path, "url", "TestBreed", "file_id")
        self.assertEqual(cat_img.breed, "TestBreed")

    def test_from_file_sets_correct_image_id(self):
        self.output_path = "test_image.png"
        cv2.imwrite(self.output_path, self.test_image_rgb)
        cat_img = CatImage.from_file(self.output_path, "url", "Breed", "file_id")
        self.assertEqual(cat_img.image_id, "file_id")

    def test_from_file_loads_correct_image_shape(self):
        self.output_path = "test_image.png"
        cv2.imwrite(self.output_path, self.test_image_rgb)
        cat_img = CatImage.from_file(self.output_path, "url", "Breed", "file_id")
        self.assertEqual(cat_img.image_data.shape, self.test_image_rgb.shape)

    # --- Filename prefix ---
    def test_get_filename_prefix_with_no_index(self):
        self.cat_image.assigned_index = None
        self.assertEqual(self.cat_image.get_filename_prefix(), "unknown_Siamese")

    def test_get_filename_prefix_with_index(self):
        self.cat_image.assigned_index = 5
        self.assertEqual(self.cat_image.get_filename_prefix(), "005_Siamese")

    def test_get_filename_prefix_with_multiword_breed(self):
        self.cat_image.assigned_index = 5
        self.cat_image._breed = "British Shorthair"
        self.assertEqual(self.cat_image.get_filename_prefix(), "005_British_Shorthair")