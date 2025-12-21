print("Проверка импорта модулей...")

try:
    from cat_image_processor import CatImageProcessor
    print("CatImageProcessor - OK")
except ImportError as e:
    print(f"CatImageProcessor - FAIL: {e}")

try:
    from cat_image_processor import CatImage
    print("CatImage - OK")
except ImportError as e:
    print(f"CatImage - FAIL: {e}")

try:
    from cat_image_processor import logger
    print("logger - OK")
except ImportError as e:
    print(f"logger - FAIL: {e}")

try:
    from cat_image_processor import ImageProcessing
    print("ImageProcessing - OK")
except ImportError as e:
    print(f"ImageProcessing - FAIL: {e}")

try:
    from cat_image_processor import implementation
    print("implementation package - OK")
except ImportError as e:
    print(f"implementation package - FAIL: {e}")

print("\nПроверка завершена!")