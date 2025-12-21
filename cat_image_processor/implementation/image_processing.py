from cat_image_processor import interfaces
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, Tuple, List
import functools


class ImageProcessing(interfaces.IImageProcessing):

    def __init__(self):
        self.kernels = {
            "blur": np.ones((3, 3)) / 9,
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        }
        self._executor = None

    @property
    def executor(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor()
        return self._executor

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)

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

    def convolution_parallel(self, image: np.ndarray, kernel_type: str, clip: bool = True,
                             num_processes: int = None) -> np.ndarray:
        if kernel_type not in self.kernels:
            raise ValueError(f"Неизвестный тип kernel: {kernel_type}. Доступные: {list(self.kernels.keys())}")

        kernel = self.kernels[kernel_type]
        if len(image.shape) == 3:
            image = self.rgb_to_grayscale(image)

        chunks = self._split_image_for_convolution(image, kernel, num_processes)

        conv_func = functools.partial(_convolution_worker, kernel=kernel, clip=clip)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(conv_func, chunks))

        return self._merge_convolution_results(results, image.shape, kernel.shape)

    def _split_image_for_convolution(self, image: np.ndarray, kernel: np.ndarray,
                                     num_processes: int = None) -> List[Dict]:
        if num_processes is None:
            num_processes = mp.cpu_count()

        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                              mode='constant', constant_values=0)

        chunks = []
        rows_per_process = max(1, img_h // num_processes)

        for i in range(num_processes):
            start_row = i * rows_per_process
            end_row = min((i + 1) * rows_per_process, img_h)

            if start_row >= img_h:
                break

            padded_start_row = start_row
            padded_end_row = end_row + kernel_h - 1

            chunk_data = {
                'image_chunk': padded_image[padded_start_row:padded_end_row],
                'global_start_row': start_row,
                'global_end_row': end_row,
                'padding': (pad_h, pad_w)
            }
            chunks.append(chunk_data)

        return chunks

    def _merge_convolution_results(self, results: List[np.ndarray],
                                   original_shape: Tuple, kernel_shape: Tuple) -> np.ndarray:
        img_h, img_w = original_shape
        kernel_h, kernel_w = kernel_shape
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        result_image = np.zeros((img_h, img_w), dtype=np.float32)

        current_row = 0
        for result in results:
            chunk_height = result.shape[0]
            result_image[current_row:current_row + chunk_height] = result
            current_row += chunk_height

        return result_image

    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
            R = image[:, :, 0]
            G = image[:, :, 1]
            B = image[:, :, 2]

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

    def gamma_correction_parallel(self, images: List[np.ndarray], gamma: float,
                                  num_processes: int = None) -> List[np.ndarray]:
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(images))

        gamma_func = functools.partial(_gamma_correction_worker, gamma=gamma)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(gamma_func, images))

        return results

    def edge_detection(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        gray = self.rgb_to_grayscale(image).astype(np.float32)

        grad_x = self.convolution(gray, "sobel_x", clip=False)
        grad_y = self.convolution(gray, "sobel_y", clip=False)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        magnitude_normalized = magnitude / magnitude.max()

        edges = np.zeros_like(magnitude_normalized)
        edges[magnitude_normalized > threshold] = 255

        return edges.astype(np.uint8)

    def edge_detection_parallel(self, image: np.ndarray, threshold: float = 0.2,
                                num_processes: int = None) -> np.ndarray:
        gray = self.rgb_to_grayscale(image).astype(np.float32)

        grad_x = self.convolution_parallel(gray, "sobel_x", clip=False, num_processes=num_processes)
        grad_y = self.convolution_parallel(gray, "sobel_y", clip=False, num_processes=num_processes)

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

    def colorful_edge_detection_parallel(self, image: np.ndarray, threshold: float = 0.2,
                                         num_processes: int = None) -> np.ndarray:
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Изображение должно быть цветным в формате RGB")

        color_image = image.astype(np.float32)

        channels = []
        for channel in range(3):
            grad_x = self.convolution_parallel(color_image[:, :, channel], "sobel_x",
                                               clip=False, num_processes=num_processes)
            grad_y = self.convolution_parallel(color_image[:, :, channel], "sobel_y",
                                               clip=False, num_processes=num_processes)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            channels.append(magnitude)

        magnitude_combined = np.maximum.reduce(channels)
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

    def process_multiple_images(self, images: List[np.ndarray], operation: str,
                                **kwargs) -> List[np.ndarray]:
        if operation == "edge_detection":
            func = functools.partial(_edge_detection_worker, **kwargs)
        elif operation == "gamma_correction":
            func = functools.partial(_gamma_correction_worker, **kwargs)
        else:
            raise ValueError(f"Неизвестная операция: {operation}")

        num_processes = min(mp.cpu_count(), len(images))

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(func, images))

        return results


def _convolution_worker(chunk_data: Dict, kernel: np.ndarray, clip: bool) -> np.ndarray:
    image_chunk = chunk_data['image_chunk']
    global_start_row = chunk_data['global_start_row']
    global_end_row = chunk_data['global_end_row']
    padding = chunk_data['padding']

    pad_h, pad_w = padding
    kernel_h, kernel_w = kernel.shape

    chunk_height = global_end_row - global_start_row
    output_chunk = np.zeros((chunk_height, image_chunk.shape[1] - 2 * pad_w), dtype=np.float32)

    for y in range(chunk_height):
        for x in range(output_chunk.shape[1]):
            region = image_chunk[y:y + kernel_h, x:x + kernel_w]
            output_chunk[y, x] = np.sum(region * kernel)

    if clip:
        output_chunk = np.clip(output_chunk, 0, 255).astype(np.uint8)

    return output_chunk


def _gamma_correction_worker(image: np.ndarray, gamma: float) -> np.ndarray:
    if image.dtype == np.uint8:
        normalized_image = image.astype(np.float32) / 255.0
        corrected_array = normalized_image ** gamma
        result = np.clip(np.round(corrected_array * 255), 0, 255).astype(np.uint8)
        return result
    else:
        return image


def _edge_detection_worker(image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    processing = ImageProcessing()
    return processing.edge_detection(image, threshold)