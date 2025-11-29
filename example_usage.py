import asyncio
import sys
import os

sys.path.append(os.path.dirname(__file__))

from implementation.cat_image_processor import CatImageProcessor
from implementation.cat_image import CatImage
from logging_config import logger


async def example_usage():
    logger.info("Запуск примера использования пакета")

    try:
        async with CatImageProcessor() as processor:
            cat_images = await processor.download_cat_images(limit=2)

            if cat_images:
                logger.info(f"Успешно скачано {len(cat_images)} изображений")

                for i, cat_image in enumerate(cat_images, 1):
                    if cat_image.assigned_index is None:
                        cat_image.assigned_index = i

                for cat_image in cat_images:
                    edges = cat_image.detect_edges_opencv()
                    grayscale = cat_image.convert_to_grayscale_opencv()
                    blurred = cat_image.apply_convolution_opencv("blur")

                    logger.info(f"Обработано изображение: {cat_image.image_id}")

                await processor.process_and_save_images(cat_images, "example_output")

            else:
                logger.warning("Не удалось скачать изображения")

    except Exception as e:
        logger.error(f"Ошибка в примере использования: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())