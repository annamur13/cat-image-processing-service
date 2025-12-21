import unittest
import os
import asyncio
from unittest.mock import patch, AsyncMock
import cv2
import numpy as np

from cat_image_processor.implementation import CatImageProcessor
from cat_image_processor.implementation import CatImage


class TestCatImageProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = CatImageProcessor()
        self.test_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def tearDown(self):
        test_files = ["https://cdn2.thecatapi.com/images/O2aNhFGU-.jpg.png  ", "test_breeds.json"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

        if os.path.exists("test_output_dir"):
            import shutil
            shutil.rmtree("test_output_dir")

    def test_initialization_output_directory(self):
        self.assertEqual(self.processor.output_directory, "cat_images_output")

    def test_initialization_client_is_none(self):
        self.assertIsNone(self.processor._client)

    @patch('httpx.AsyncClient')
    def test_context_manager_assigns_client(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        async def run():
            async with CatImageProcessor() as processor:
                return processor._client

        client = asyncio.run(run())
        self.assertEqual(client, mock_client_instance)

    @patch('httpx.AsyncClient')
    def test_successful_api_call_returns_two_images(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response_search = AsyncMock()
        mock_response_search.status_code = 200
        mock_response_search.json.return_value = [
            {"id": "abc123", "url": "https://cdn2.thecatapi.com/images/O2aNhFGU-.jpg  ", "breeds": [{"name": "Siamese", "id": "siam"}]},
            {"id": "def456", "url": "https://cdn2.thecatapi.com/images/O2aNhFGU-.jpg  ", "breeds": [{"name": "Persian", "id": "pers"}]}
        ]

        mock_response_image = AsyncMock()
        mock_response_image.status_code = 200
        mock_response_image.content = cv2.imencode('.jpg', self.test_image_data)[1].tobytes()

        mock_client_instance.get.side_effect = [mock_response_search, mock_response_image, mock_response_image]

        async def run():
            self.processor._client = mock_client_instance
            return await self.processor.download_cat_images(limit=2)

        images = asyncio.run(run())
        self.assertEqual(len(images), 2)

    @patch('httpx.AsyncClient')
    def test_successful_api_call_first_image_id(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response_search = AsyncMock()
        mock_response_search.status_code = 200
        mock_response_search.json.return_value = [
            {"id": "123", "url": "...", "breeds": [{"name": "Siamese", "id": "siam"}]}
        ]
        mock_response_image = AsyncMock()
        mock_response_image.status_code = 200
        mock_response_image.content = cv2.imencode('.jpg', self.test_image_data)[1].tobytes()
        mock_client_instance.get.side_effect = [mock_response_search, mock_response_image]

        async def run():
            self.processor._client = mock_client_instance
            images = await self.processor.download_cat_images(limit=1)
            return images[0].image_id

        image_id = asyncio.run(run())
        self.assertEqual(image_id, "123")

    @patch('httpx.AsyncClient')
    def test_successful_api_call_first_breed(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response_search = AsyncMock()
        mock_response_search.status_code = 200
        mock_response_search.json.return_value = [
            {"id": "123", "url": "...", "breeds": [{"name": "Siamese", "id": "siam"}]}
        ]
        mock_response_image = AsyncMock()
        mock_response_image.status_code = 200
        mock_response_image.content = cv2.imencode('.jpg', self.test_image_data)[1].tobytes()
        mock_client_instance.get.side_effect = [mock_response_search, mock_response_image]

        async def run():
            self.processor._client = mock_client_instance
            images = await self.processor.download_cat_images(limit=1)
            return images[0].breed

        breed = asyncio.run(run())
        self.assertEqual(breed, "Siamese")

    @patch('httpx.AsyncClient')
    def test_api_call_with_breed_sends_correct_param(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response_breeds = AsyncMock()
        mock_response_breeds.status_code = 200
        mock_response_breeds.json.return_value = [{"id": "siam", "name": "Siamese"}]

        mock_response_search = AsyncMock()
        mock_response_search.status_code = 200
        mock_response_search.json.return_value = [{"id": "1", "url": "...", "breeds": [{"name": "Siamese", "id": "siam"}]}]

        mock_response_image = AsyncMock()
        mock_response_image.status_code = 200
        mock_response_image.content = b"fake image"

        mock_client_instance.get.side_effect = [mock_response_breeds, mock_response_search, mock_response_image]

        async def run():
            self.processor._client = mock_client_instance
            await self.processor.download_cat_images(limit=1, breed="Siamese")
            return mock_client_instance.get.call_args_list[1][1]['params'].get('breed_ids')

        breed_id_param = asyncio.run(run())
        self.assertEqual(breed_id_param, 'siam')

    @patch('httpx.AsyncClient')
    def test_api_call_failure_returns_empty_list(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server Error")
        mock_client_instance.get.return_value = mock_response

        async def run():
            self.processor._client = mock_client_instance
            return await self.processor.download_cat_images(limit=1)

        images = asyncio.run(run())
        self.assertEqual(len(images), 0)

    @patch('aiofiles.open')
    @patch('cv2.imencode')
    def test_save_image_calls_aiofiles_once(self, mock_imencode, mock_aiofiles):
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_file = AsyncMock()
        mock_aiofiles.return_value.__aenter__.return_value = mock_file

        async def run():
            await self.processor._save_image_with_aiofiles("test_output.png", self.test_image_data)

        asyncio.run(run())
        self.assertEqual(mock_aiofiles.call_count, 1)

    @patch('cat_image_processor.implementation.cat_image_processor.CatImageProcessor._save_image_with_aiofiles')
    def test_process_and_save_images_calls_save_six_times(self, mock_save):
        test_images = []
        for i in range(2):
            cat_img = CatImage(image_data=self.test_image_data, url=f"url{i}", breed="Siamese", image_id=f"id{i}")
            cat_img.assigned_index = i + 1
            test_images.append(cat_img)

        mock_save.return_value = None

        async def run():
            await self.processor.process_and_save_images(test_images, "test_output_dir")

        asyncio.run(run())
        self.assertEqual(mock_save.call_count, 6)

    def test_output_directory_property_returns_default(self):
        self.assertEqual(self.processor.output_directory, "cat_images_output")

    def test_output_directory_setter_updates_value(self):
        self.processor.output_directory = "new_output_dir"
        self.assertEqual(self.processor._output_dir, "new_output_dir")

    @patch('httpx.AsyncClient')
    def test_get_breed_id_success_returns_pers(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "siam", "name": "Siamese"},
            {"id": "pers", "name": "Persian"}
        ]
        mock_client_instance.get.return_value = mock_response

        async def run():
            self.processor._client = mock_client_instance
            return await self.processor._get_breed_id("Persian")

        breed_id = asyncio.run(run())
        self.assertEqual(breed_id, "pers")

    @patch('httpx.AsyncClient')
    def test_get_breed_id_not_found_returns_none(self, mock_client):
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "siam", "name": "Siamese"}]
        mock_client_instance.get.return_value = mock_response

        async def run():
            self.processor._client = mock_client_instance
            return await self.processor._get_breed_id("Nonexistent")

        breed_id = asyncio.run(run())
        self.assertIsNone(breed_id)


class TestCatImageProcessorIntegration(unittest.TestCase):

    def setUp(self):
        self.processor = CatImageProcessor()

    @unittest.skipIf(not os.getenv('CAT_API_KEY'), "Требуется API ключ для интеграционных тестов")
    def test_real_api_call_returns_one_image(self):
        async def run():
            async with self.processor:
                images = await self.processor.download_cat_images(limit=1)
                return len(images)

        count = asyncio.run(run())
        self.assertEqual(count, 1)

    @unittest.skipIf(not os.getenv('CAT_API_KEY'), "Требуется API ключ для интеграционных тестов")
    def test_real_api_call_first_image_is_catimage(self):
        async def run():
            async with self.processor:
                images = await self.processor.download_cat_images(limit=1)
                return isinstance(images[0], CatImage)

        is_catimage = asyncio.run(run())
        self.assertTrue(is_catimage)

    @unittest.skipIf(not os.getenv('CAT_API_KEY'), "Требуется API ключ для интеграционных тестов")
    def test_real_api_call_image_has_data(self):
        async def run():
            async with self.processor:
                images = await self.processor.download_cat_images(limit=1)
                return images[0].image_data is not None

        has_data = asyncio.run(run())
        self.assertTrue(has_data)


if __name__ == '__main__':
    unittest.main()