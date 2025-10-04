import cv2
import interfaces
import numpy as np

class OpenCVImageProcessing(interfaces.IImageProcessing):

    def convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
            return result
        else:
            return cv2.filter2D(image, -1, kernel)

    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:

        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:

            return image

    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:

        image_normalized = image.astype(np.float32) / 255.0

        corrected = np.power(image_normalized, gamma)

        return (corrected * 255).astype(np.uint8)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:

        if len(image.shape) == 3:
            gray = self.rgb_to_grayscale(image)
        else:
            gray = image

        edges = cv2.Canny(gray, 100, 200)

        return edges

    def corner_detection(self, image: np.ndarray,
                         max_corners: int = 100,
                         quality_level: float = 0.01,
                         min_distance: float = 10.0) -> np.ndarray:

        if len(image.shape) == 3:
            gray = self.rgb_to_grayscale(image)
        else:
            gray = image

        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)

        result = image.copy()
        if corners is not None:
            corners = corners.astype(np.int32)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(result, (x, y), 5, (0, 255, 0), -1)

        return result

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5,
                      sigma_x: float = 0) -> np.ndarray:

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)

    def sobel_edge_detection(self, image: np.ndarray,
                             dx: int = 1, dy: int = 1,
                             ksize: int = 3) -> np.ndarray:

        if len(image.shape) == 3:
            gray = self.rgb_to_grayscale(image)
        else:
            gray = image

        sobelx = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return magnitude