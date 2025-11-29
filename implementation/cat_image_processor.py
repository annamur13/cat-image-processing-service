import httpx
import aiofiles
import asyncio
import os
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from logging_config import logger
from implementation.cat_image import CatImage
from Utils.timer_decorator import timer_decorator
import config


class CatImageProcessor:

    def __init__(self, api_key: str = config.CAT_API_KEY, base_url: str = config.CAT_API_BASE_URL):
        self._api_key = api_key
        self._base_url = base_url
        self._output_dir = "cat_images_output"
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    @timer_decorator
    async def download_cat_images(self, limit: int = 1, breed: Optional[str] = None) -> List[CatImage]:
        print(f"🐱 Downloading {limit} cat images{'' if not breed else ' of breed: ' + breed}")

        try:
            url = f"{self._base_url}/images/search"
            params = {
                'limit': limit,
                'size': 'med',
                'mime_types': 'jpg,png'
            }

            if breed:
                breed_id = await self._get_breed_id(breed)
                print(f"Breed ID for {breed}: {breed_id}")
                if breed_id:
                    params['breed_ids'] = breed_id

            headers = {"x-api-key": self._api_key} if self._api_key else {}

            response = await self._client.get(url, params=params, headers=headers)

            response.raise_for_status()
            data = response.json()

            download_tasks = [self._download_single_image(img_data) for img_data in data]
            cat_images = await asyncio.gather(*download_tasks)

            cat_images = [img for img in cat_images if img is not None]

            print(f"Successfully downloaded {len(cat_images)} images")
            return cat_images

        except httpx.HTTPError as e:
            print(f"HTTP Error downloading images: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in download_cat_images: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _get_breed_id(self, breed_name: str) -> Optional[str]:
        try:
            url = f"{self._base_url}/breeds"
            headers = {"x-api-key": self._api_key} if self._api_key else {}

            response = await self._client.get(url, headers=headers)

            response.raise_for_status()
            breeds = response.json()

            for breed in breeds:
                if breed['name'].lower() == breed_name.lower():
                    return breed['id']

            return None

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    async def _download_single_image(self, img_data: Dict[str, Any]) -> Optional[CatImage]:
        try:
            image_url = img_data['url']
            image_id = img_data['id']

            breed_info = img_data.get('breeds', [{}])[0] if img_data.get('breeds') else {}
            breed_name = breed_info.get('name', 'unknown')

            response = await self._client.get(image_url)
            response.raise_for_status()
            image_content = response.content
            print(f"Downloaded {len(image_content)} bytes")

            img_array = np.frombuffer(image_content, np.uint8)
            image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image_data is None:
                print(f"Could not decode image {image_id}")
                return None

            print(f"Successfully decoded image {image_id}, shape: {image_data.shape}")
            return CatImage(image_data, image_url, breed_name, image_id)

        except Exception as e:
            print(f"Error downloading single image {img_data.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return None

    @timer_decorator
    async def process_and_save_images(self, cat_images: List[CatImage], output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            output_dir = self._output_dir

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving images to: {output_dir}")

        save_tasks = []
        for cat_image in cat_images:
            task = self._save_single_image(cat_image, output_dir)
            save_tasks.append(task)

        await asyncio.gather(*save_tasks)

    async def _save_single_image(self, cat_image: CatImage, output_dir: str) -> None:
        try:
            index = cat_image.assigned_index
            print(f"\n--- Processing image {index} ---")

            if index is None:
                index_str = cat_image.image_id or "unknown"
                prefix = f"{index_str}_{cat_image.breed.replace(' ', '_')}"
            else:
                prefix = f"{index:03d}_{cat_image.breed.replace(' ', '_')}"

            original_filename = f"{prefix}_original.png"
            original_path = os.path.join(output_dir, original_filename)
            await self._save_image_with_aiofiles(original_path, cat_image.image_data)

            manual_edges = cat_image.detect_edges_manual()
            manual_filename = f"{prefix}_manual_edges.png"
            manual_path = os.path.join(output_dir, manual_filename)
            await self._save_image_with_aiofiles(manual_path, manual_edges)

            opencv_edges = cat_image.detect_edges_opencv()
            opencv_filename = f"{prefix}_opencv_edges.png"
            opencv_path = os.path.join(output_dir, opencv_filename)
            await self._save_image_with_aiofiles(opencv_path, opencv_edges)

            print(f"Saved: {original_filename}")
            print(f"Saved: {manual_filename}")
            print(f"Saved: {opencv_filename}")

        except Exception as e:
            print(f"Error processing image {cat_image.assigned_index}: {e}")
            import traceback
            traceback.print_exc()

    async def _save_image_with_aiofiles(self, filepath: str, image_data: np.ndarray) -> None:
        try:
            success, encoded_image = cv2.imencode('.png', image_data)
            if not success:
                raise Exception("Failed to encode image")

            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(encoded_image.tobytes())

        except Exception as e:
            print(f"Error saving image {filepath}: {e}")
            raise

    @property
    def output_directory(self) -> str:
        return self._output_dir

    @output_directory.setter
    def output_directory(self, value: str) -> None:
        self._output_dir = value