import interfaces

import numpy as np
from scipy.ndimage import maximum_filter

class ImageProcessing(interfaces.IImageProcessing):

    def __init__(self):

        self.kernels = {
            "blur": np.ones((3, 3)) / 9,
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }

    def convolution(self, image: np.ndarray, kernel_type: str, clip: bool = True) -> np.ndarray:
        if kernel_type not in self.kernels:
            raise ValueError(f"Неизвестный тип kernel: {kernel_type}. Доступные: {list(self.kernels.keys())}")

        kernel = self.kernels[kernel_type]
        if len(image.shape) == 3:
            image = self.rgb_to_grayscale(image)
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape

        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        output = np.zeros_like(image, dtype=np.float32)
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                              mode='constant', constant_values=0)

        for y in range(img_h):
            for x in range(img_w):
                region = padded_image[y:y + kernel_h, x:x + kernel_w]
                output[y, x] = np.sum(region * kernel)

        if clip:
            output = np.clip(output, 0, 255).astype(np.uint8)

        return output

    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
            R = image[:,:, 0]
            G = image[:,:, 1]
            B = image[:,:, 2]

            gray = 0.299 * R + 0.587 * G + 0.114 * B
            gray = np.round(gray)
            gray = np.clip(gray, 0, 255)
            gray = gray.astype(np.uint8)
            return gray
        else:
            print("Ошибка: некорретный тип изображения или изображение уже черно-белое")
            return image

    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        if image.dtype == np.uint8:
            normalized_image = image.astype(np.float32) / 255.0
            corrected_array = normalized_image ** gamma
            result = np.clip(np.round(corrected_array * 255), 0, 255).astype(np.uint8)
            return result
        else:
            print("Ошибка: некорректный тип изображения")
            return image

    def edge_detection(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        gray = self.rgb_to_grayscale(image).astype(np.float32)

        grad_x = self.convolution(gray, "sobel_x", clip=False)
        grad_y = self.convolution(gray, "sobel_y", clip=False)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        magnitude_normalized = magnitude / magnitude.max()

        edges = np.zeros_like(magnitude_normalized)
        edges[magnitude_normalized > threshold] = 255

        return edges.astype(np.uint8)

    def colorful_edge_detection(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Изображение должно быть цветным в формате RGB")

        color_image = image.astype(np.float32)

        grad_x_r = self.convolution(color_image[:, :, 0], "sobel_x", clip=False)
        grad_y_r = self.convolution(color_image[:, :, 0], "sobel_y", clip=False)

        grad_x_g = self.convolution(color_image[:, :, 1], "sobel_x", clip=False)
        grad_y_g = self.convolution(color_image[:, :, 1], "sobel_y", clip=False)

        grad_x_b = self.convolution(color_image[:, :, 2], "sobel_x", clip=False)
        grad_y_b = self.convolution(color_image[:, :, 2], "sobel_y", clip=False)

        magnitude_r = np.sqrt(grad_x_r ** 2 + grad_y_r ** 2)
        magnitude_g = np.sqrt(grad_x_g ** 2 + grad_y_g ** 2)
        magnitude_b = np.sqrt(grad_x_b ** 2 + grad_y_b ** 2)

        magnitude_combined = np.maximum.reduce([magnitude_r, magnitude_g, magnitude_b])
        magnitude_normalized = magnitude_combined / magnitude_combined.max()

        edges_color = np.zeros_like(color_image)

        edge_mask = magnitude_normalized > threshold
        edges_color[edge_mask] = color_image[edge_mask]

        return edges_color.astype(np.uint8)

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        gray = self.rgb_to_grayscale(image).astype(np.float32)

        Ix = self.convolution(gray, "sobel_x", clip=False)
        Iy = self.convolution(gray, "sobel_y", clip=False)

        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        Sxx = self.convolution(Ixx, "blur", clip=False)
        Sxy = self.convolution(Ixy, "blur", clip=False)
        Syy = self.convolution(Iyy, "blur", clip=False)

        det = Sxx * Syy - Sxy ** 2
        trace = Sxx + Syy
        k = 0.04
        harris_response = det - k * (trace ** 2)

        threshold = 0.01 * harris_response.max()

        from scipy.ndimage import maximum_filter
        neighborhood_size = 5
        local_max = maximum_filter(harris_response, size=neighborhood_size) == harris_response
        corner_mask = (harris_response > threshold) & local_max

        result = image.copy() if image.ndim == 3 else np.dstack([gray.astype(np.uint8)] * 3)
        ys, xs = np.where(corner_mask)

        for y, x in zip(ys, xs):
            if 3 <= y < result.shape[0] - 3 and 3 <= x < result.shape[1] - 3:

                size = 3
                color = [0, 255, 0]
                result[y - size:y + size + 1, x] = color
                result[y, x - size:x + size + 1] = color

        return result