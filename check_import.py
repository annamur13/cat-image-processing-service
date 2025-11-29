import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("Проверка импорта модулей...")

try:
    from implementation.cat_image_processor import CatImageProcessor
    print("CatImageProcessor - OK")
except ImportError as e:
    print(f"CatImageProcessor - FAIL: {e}")

try:
    from implementation.cat_image import CatImage
    print("CatImage - OK")
except ImportError as e:
    print(f"CatImage - FAIL: {e}")

try:
    from logging_config import logger
    print("logger - OK")
except ImportError as e:
    print(f"logger - FAIL: {e}")

try:
    from implementation.image_processing import ImageProcessing
    print("ImageProcessing - OK")
except ImportError as e:
    print(f"ImageProcessing - FAIL: {e}")

try:
    import implementation
    print("implementation package - OK")
except ImportError as e:
    print(f"implementation package - FAIL: {e}")

print("\nПроверка завершена!")