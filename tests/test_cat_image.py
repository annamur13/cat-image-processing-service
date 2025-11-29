import unittest
import os
import cv2
import numpy as np
from pathlib import Path
from implementation.cat_image import CatImage


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

    def test_properties(self):
        self.assertEqual(self.cat_image.url, "https://cdn2.thecatapi.com/images/O2aNhFGU-.jpg")
        self.assertEqual(self.cat_image.breed, "Siamese")
        self.assertEqual(self.cat_image.image_id, "test_001")
        np.testing.assert_array_equal(self.cat_image.image_data, self.test_image_rgb)

    def test_detect_edges_manual(self):
        edges = self.cat_image.detect_edges_manual(threshold=0.2)
        self.assertEqual(edges.shape, self.test_image_rgb.shape[:2])
        self.assertEqual(edges.dtype, np.uint8)

    def test_detect_edges_opencv(self):
        edges = self.cat_image.detect_edges_opencv()
        self.assertEqual(len(edges.shape), 2)
        self.assertEqual(edges.shape, self.test_image_rgb.shape[:2])
        self.assertEqual(edges.dtype, np.uint8)

    def test_apply_convolution_manual(self):
        for kernel_type in ["blur", "sharpen", "sobel_x"]:
            with self.subTest(kernel_type=kernel_type):
                result = self.cat_image.apply_convolution_manual(kernel_type)
                expected_shape = self.test_image_rgb.shape[:2]
                self.assertEqual(result.shape, expected_shape)
                self.assertEqual(result.dtype, np.uint8)

    def test_apply_convolution_opencv(self):
        for kernel_type in ["blur", "sharpen", "sobel"]:
            with self.subTest(kernel_type=kernel_type):
                result = self.cat_image.apply_convolution_opencv(kernel_type)
                if kernel_type == "sobel":
                    expected_shape = self.test_image_rgb.shape[:2]
                else:
                    expected_shape = self.test_image_rgb.shape
                self.assertEqual(result.shape, expected_shape)
                self.assertEqual(result.dtype, np.uint8)

    def test_convert_to_grayscale(self):
        grayscale_manual = self.cat_image.convert_to_grayscale_manual()
        self.assertEqual(len(grayscale_manual.shape), 2)
        self.assertEqual(grayscale_manual.dtype, np.uint8)

        grayscale_opencv = self.cat_image.convert_to_grayscale_opencv()
        self.assertEqual(len(grayscale_opencv.shape), 2)
        self.assertEqual(grayscale_opencv.dtype, np.uint8)

    def test_image_addition(self):
        result = self.cat_image + self.cat_image2
        self.assertIsInstance(result, CatImage)
        self.assertEqual(result.breed, "Siamese")
        self.assertTrue("combined" in result.url)
        self.assertEqual(result.image_data.shape, self.test_image_rgb.shape)

    def test_image_subtraction(self):
        result = self.cat_image - self.cat_image2
        self.assertIsInstance(result, CatImage)
        self.assertEqual(result.breed, "Siamese")
        self.assertTrue("subtracted" in result.url)
        self.assertEqual(result.image_data.shape, self.test_image_rgb.shape)

    def test_addition_with_different_sizes_raises_error(self):
        small_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cat_image_small = CatImage(
            small_image,
            "https://cdn2.thecatapi.com/images/HD4lZB6BI.jpg",
            "Maine Coon",
            "small_001"
        )
        with self.assertRaises(ValueError):
            _ = self.cat_image + cat_image_small

    def test_from_file_classmethod(self):
        self.output_path = "test_image.png"
        cv2.imwrite(self.output_path, self.test_image_rgb)

        cat_image_from_file = CatImage.from_file(
            self.output_path,
            "https://cdn2.thecatapi.com/images/J2PmlIizw.jpg",
            "Bengal",
            "file_001"
        )
        self.assertEqual(cat_image_from_file.url, "https://cdn2.thecatapi.com/images/J2PmlIizw.jpg")
        self.assertEqual(cat_image_from_file.breed, "Bengal")
        self.assertEqual(cat_image_from_file.image_id, "file_001")
        self.assertEqual(cat_image_from_file.image_data.shape, self.test_image_rgb.shape)

    def test_get_filename_prefix(self):
        self.cat_image.assigned_index = None
        prefix = self.cat_image.get_filename_prefix()
        self.assertEqual(prefix, "unknown_Siamese")

        self.cat_image.assigned_index = 5
        prefix = self.cat_image.get_filename_prefix()
        self.assertEqual(prefix, "005_Siamese")

        self.cat_image._breed = "British Shorthair"
        prefix = self.cat_image.get_filename_prefix()
        self.assertEqual(prefix, "005_British_Shorthair")


if __name__ == '__main__':
    unittest.main()