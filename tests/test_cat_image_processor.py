import unittest
import asyncio
import os
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import cv2
import numpy as np

from implementation.cat_image_processor import CatImageProcessor
from implementation.cat_image import CatImage


class TestCatImageProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = CatImageProcessor()
        self.test_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def tearDown(self):
        test_files = ["test_output.png", "test_breeds.json"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

        if os.path.exists("test_output_dir"):
            import shutil
            shutil.rmtree("test_output_dir")

    def test_initialization(self):
        self.assertEqual(self.processor.output_directory, "cat_images_output")
        self.assertIsNone(self.processor._client)

    @patch('httpx.AsyncClient')
    async def test_context_manager(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        async with CatImageProcessor() as processor:
            self.assertEqual(processor._client, mock_client_instance)

        mock_client_instance.aclose.assert_called_once()

    @patch('implementation.cat_image_processor.httpx.AsyncClient')
    async def test_successful_api_call(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response_search = AsyncMock()
        mock_response_search.status_code = 200
        mock_response_search.json.return_value = [
            {
                "id": "abc123",
                "url": "https://cdn2.thecatapi.com/images/123.jpg",
                "breeds": [{"name": "Siamese", "id": "siam"}]
            },
            {
                "id": "def456",
                "url": "https://cdn2.thecatapi.com/images/456.jpg",
                "breeds": [{"name": "Persian", "id": "pers"}]
            }
        ]

        mock_response_image = AsyncMock()
        mock_response_image.status_code = 200
        mock_response_image.content = cv2.imencode('.jpg', self.test_image_data)[1].tobytes()

        mock_client_instance.get.side_effect = [
            mock_response_search,
            mock_response_image,
            mock_response_image
        ]

        self.processor._client = mock_client_instance

        images = await self.processor.download_cat_images(limit=2)

        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].image_id, "123")
        self.assertEqual(images[0].breed, "Siamese")
        self.assertEqual(images[1].image_id, "456")
        self.assertEqual(images[1].breed, "Persian")

        self.assertIsNotNone(images[0].image_data)
        self.assertIsNotNone(images[1].image_data)

    @patch('implementation.cat_image_processor.httpx.AsyncClient')
    async def test_api_call_with_breed(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response_breeds = AsyncMock()
        mock_response_breeds.status_code = 200
        mock_response_breeds.json.return_value = [
            {"id": "siam", "name": "Siamese"},
            {"id": "pers", "name": "Persian"}
        ]

        mock_response_search = AsyncMock()
        mock_response_search.status_code = 200
        mock_response_search.json.return_value = [
            {
                "id": "1",
                "url": "https://cdn2.thecatapi.com/images/1.jpg",
                "breeds": [{"name": "Siamese", "id": "siam"}]
            }
        ]

        mock_response_image = AsyncMock()
        mock_response_image.status_code = 200
        mock_response_image.content = cv2.imencode('.jpg', self.test_image_data)[1].tobytes()

        mock_client_instance.get.side_effect = [
            mock_response_breeds,
            mock_response_search,
            mock_response_image
        ]

        self.processor._client = mock_client_instance

        images = await self.processor.download_cat_images(limit=1, breed="Siamese")

        call_args = mock_client_instance.get.call_args_list[1]
        params = call_args[1]['params']  # kwargs
        self.assertEqual(params.get('breed_ids'), 'siam')

    @patch('implementation.cat_image_processor.httpx.AsyncClient')
    async def test_api_call_failure(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server Error")

        mock_client_instance.get.return_value = mock_response
        self.processor._client = mock_client_instance

        images = await self.processor.download_cat_images(limit=1)

        self.assertEqual(len(images), 0)

    @patch('aiofiles.open')
    @patch('cv2.imencode')
    async def test_save_image_with_aiofiles(self, mock_imencode, mock_aiofiles):
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        mock_file = AsyncMock()
        mock_aiofiles.return_value.__aenter__.return_value = mock_file

        await self.processor._save_image_with_aiofiles("test_output.png", self.test_image_data)

        mock_aiofiles.assert_called_once_with("test_output.png", 'wb')
        mock_file.write.assert_called_once()

    @patch('implementation.cat_image_processor.CatImageProcessor._save_image_with_aiofiles')
    async def test_process_and_save_images(self, mock_save):
        test_images = []
        for i in range(2):
            cat_image = CatImage(
                image_data=self.test_image_data,
                url=f"https://cdn2.thecatapi.com/images/test{i}.jpg",
                breed="Siamese",
                image_id=f"test_{i}"
            )
            cat_image.assigned_index = i + 1
            test_images.append(cat_image)

        mock_save.return_value = None

        output_dir = "test_output_dir"
        await self.processor.process_and_save_images(test_images, output_dir)

        self.assertEqual(mock_save.call_count, 6)

    def test_output_directory_property(self):
        self.assertEqual(self.processor.output_directory, "cat_images_output")

        self.processor.output_directory = "new_output_dir"
        self.assertEqual(self.processor.output_directory, "new_output_dir")
        self.assertEqual(self.processor._output_dir, "new_output_dir")

    @patch('implementation.cat_image_processor.httpx.AsyncClient')
    async def test_get_breed_id_success(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "siam", "name": "Siamese"},
            {"id": "pers", "name": "Persian"},
            {"id": "beng", "name": "Bengal"}
        ]

        mock_client_instance.get.return_value = mock_response
        self.processor._client = mock_client_instance

        breed_id = await self.processor._get_breed_id("Persian")

        self.assertEqual(breed_id, "pers")

    @patch('implementation.cat_image_processor.httpx.AsyncClient')
    async def test_get_breed_id_not_found(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "siam", "name": "Siamese"},
            {"id": "pers", "name": "Persian"}
        ]

        mock_client_instance.get.return_value = mock_response
        self.processor._client = mock_client_instance

        breed_id = await self.processor._get_breed_id("Nonexistent")

        self.assertIsNone(breed_id)


class TestCatImageProcessorIntegration(unittest.TestCase):

    def setUp(self):
        self.processor = CatImageProcessor()

    @unittest.skipIf(not os.getenv('CAT_API_KEY'), "Требуется API ключ для интеграционных тестов")
    async def test_real_api_call(self):
        async with self.processor:
            images = await self.processor.download_cat_images(limit=1)

            self.assertEqual(len(images), 1)
            self.assertIsInstance(images[0], CatImage)
            self.assertIsNotNone(images[0].image_data)
            self.assertIsNotNone(images[0].url)
            self.assertIsNotNone(images[0].breed)


if __name__ == '__main__':
    unittest.main()