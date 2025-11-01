import requests
import os
from typing import List, Optional, Dict, Any
import cv2
import numpy as np

from implementation.cat_image import AbstractCatImage, ColorCatImage, GrayscaleCatImage
from Utils.timer_decorator import timer_decorator
import config


class CatImageProcessor:

    def __init__(self, api_key: str = config.CAT_API_KEY, base_url: str = config.CAT_API_BASE_URL):
        self._api_key = api_key
        self._base_url = base_url
        self._output_dir = "cat_images_output"

    @timer_decorator
    def download_cat_images(self, limit: int = 1, breed: Optional[str] = None,
                            image_type: str = "color") -> List[AbstractCatImage]:

        print(f"🐱 Downloading {limit} {image_type} cat images{'' if not breed else ' of breed: ' + breed}")

        try:
            url = f"{self._base_url}/images/search"
            params = {
                'limit': limit,
                'size': 'med',
                'mime_types': 'jpg,png'
            }

            if breed:
                breed_id = self._get_breed_id(breed)
                if breed_id:
                    params['breed_ids'] = breed_id

            headers = {"x-api-key": self._api_key} if self._api_key else {}
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            cat_images = []
            for img_data in data:
                cat_image = self._download_single_image(img_data, image_type)
                if cat_image:
                    cat_images.append(cat_image)

            print(f"Successfully downloaded {len(cat_images)} {image_type} images")
            return cat_images

        except requests.exceptions.RequestException as e:
            print(f"Error downloading images: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def _get_breed_id(self, breed_name: str) -> Optional[str]:
        try:
            url = f"{self._base_url}/breeds"
            headers = {"x-api-key": self._api_key} if self._api_key else {}
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            breeds = response.json()
            for breed in breeds:
                if breed['name'].lower() == breed_name.lower():
                    return breed['id']

            print(f"Breed '{breed_name}' not found. Available breeds: {[b['name'] for b in breeds][:5]}...")
            return None

        except Exception as e:
            print(f"Error getting breed ID: {e}")
            return None

    def _download_single_image(self, img_data: Dict[str, Any], image_type: str = "color") -> Optional[AbstractCatImage]:
        try:
            image_url = img_data['url']
            image_id = img_data['id']

            breed_info = img_data.get('breeds', [{}])[0] if img_data.get('breeds') else {}
            breed_name = breed_info.get('name', 'unknown')

            print(f"Downloading {image_type} image {image_id} ({breed_name}) from {image_url}")

            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            img_array = np.frombuffer(img_response.content, np.uint8)

            if image_type.lower() == "grayscale":
                image_data = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                image_class = GrayscaleCatImage
            else:
                image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                image_class = ColorCatImage

            if image_data is None:
                print(f"Could not decode image {image_id}")
                return None

            return image_class(image_data, image_url, breed_name, image_id)

        except Exception as e:
            print(f"Error downloading single image: {e}")
            return None

    @timer_decorator
    def process_and_save_images(self, cat_images: List[AbstractCatImage],
                                output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            output_dir = self._output_dir

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving images to: {output_dir}")

        for i, cat_image in enumerate(cat_images, 1):
            print(f"\n--- Processing image {i}/{len(cat_images)}: {cat_image.breed} ({type(cat_image).__name__}) ---")

            try:
                original_filename = f"{i}_{cat_image.breed.replace(' ', '_')}_{type(cat_image).__name__}_original.png"
                original_path = os.path.join(output_dir, original_filename)
                cv2.imwrite(original_path, cat_image.image_data)

                manual_edges = cat_image.detect_edges_manual()
                manual_filename = f"{i}_{cat_image.breed.replace(' ', '_')}_{type(cat_image).__name__}_manual_edges.png"
                manual_path = os.path.join(output_dir, manual_filename)
                cv2.imwrite(manual_path, manual_edges)

                opencv_edges = cat_image.detect_edges_opencv()
                opencv_filename = f"{i}_{cat_image.breed.replace(' ', '_')}_{type(cat_image).__name__}_opencv_edges.png"
                opencv_path = os.path.join(output_dir, opencv_filename)
                cv2.imwrite(opencv_path, opencv_edges)

                blur_result = cat_image.apply_convolution_opencv("blur")
                blur_filename = f"{i}_{cat_image.breed.replace(' ', '_')}_{type(cat_image).__name__}_blur.png"
                blur_path = os.path.join(output_dir, blur_filename)
                cv2.imwrite(blur_path, blur_result)

                print(f"Saved: {original_filename}")
                print(f"Saved: {manual_filename}")
                print(f"Saved: {opencv_filename}")
                print(f"Saved: {blur_filename}")

            except Exception as e:
                print(f"Error processing image {i}: {e}")

    @timer_decorator
    def process_multiple_breeds(self, breeds: List[str], images_per_breed: int = 2,
                                image_type: str = "color") -> None:
        print(f"Starting batch processing for {len(breeds)} breeds ({image_type} images)")

        all_images = []
        for breed in breeds:
            print(f"\nProcessing breed: {breed}")
            breed_images = self.download_cat_images(limit=images_per_breed, breed=breed, image_type=image_type)
            all_images.extend(breed_images)

        if all_images:
            breed_dir = f"{self._output_dir}_multiple_breeds_{image_type}"
            self.process_and_save_images(all_images, breed_dir)
        else:
            print("No images were downloaded")

    def convert_images_to_grayscale(self, color_images: List[ColorCatImage]) -> List[GrayscaleCatImage]:
        grayscale_images = []
        for color_image in color_images:
            try:
                gray_image = GrayscaleCatImage.from_color_image(color_image)
                grayscale_images.append(gray_image)
            except Exception as e:
                print(f"Error converting image to grayscale: {e}")

        return grayscale_images

    def compare_processing_methods(self, cat_image: AbstractCatImage) -> Dict[str, Any]:
        results = {
            'image_type': type(cat_image).__name__,
            'breed': cat_image.breed,
            'shape': cat_image.image_data.shape
        }

        try:
            import time

            start_time = time.time()
            manual_edges = cat_image.detect_edges_manual()
            manual_time = time.time() - start_time

            start_time = time.time()
            opencv_edges = cat_image.detect_edges_opencv()
            opencv_time = time.time() - start_time

            results['edge_detection'] = {
                'manual_time': manual_time,
                'opencv_time': opencv_time,
                'speedup': manual_time / opencv_time if opencv_time > 0 else float('inf')
            }

            if isinstance(cat_image, ColorCatImage):
                start_time = time.time()
                manual_gray = cat_image.convert_to_grayscale_manual()
                manual_gray_time = time.time() - start_time

                start_time = time.time()
                opencv_gray = cat_image.convert_to_grayscale_opencv()
                opencv_gray_time = time.time() - start_time

                results['grayscale_conversion'] = {
                    'manual_time': manual_gray_time,
                    'opencv_time': opencv_gray_time,
                    'speedup': manual_gray_time / opencv_gray_time if opencv_gray_time > 0 else float('inf')
                }

            return results

        except Exception as e:
            print(f"Error in comparison: {e}")
            return results

    @staticmethod
    def get_available_breeds(api_key: str = config.CAT_API_KEY) -> List[str]:
        try:
            url = "https://api.thecatapi.com/v1/breeds"
            headers = {"x-api-key": api_key} if api_key else {}
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            breeds = response.json()
            return [breed['name'] for breed in breeds]

        except Exception as e:
            print(f"Error getting available breeds: {e}")
            return []

    @property
    def output_directory(self) -> str:
        return self._output_dir

    @output_directory.setter
    def output_directory(self, value: str) -> None:
        self._output_dir = value

def demo_processor():
    processor = CatImageProcessor()

    print("=== Downloading color images ===")
    color_images = processor.download_cat_images(limit=2, breed="Siamese", image_type="color")

    print("\n=== Downloading grayscale images ===")
    gray_images = processor.download_cat_images(limit=2, breed="Persian", image_type="grayscale")

    print("\n=== Converting color to grayscale ===")
    converted_gray = processor.convert_images_to_grayscale(color_images)

    all_images = color_images + gray_images + converted_gray
    processor.process_and_save_images(all_images)

    if color_images:
        comparison = processor.compare_processing_methods(color_images[0])
        print(f"\nProcessing comparison: {comparison}")

if __name__ == "__main__":
    demo_processor()