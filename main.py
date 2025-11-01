import argparse
import os
import time
import cv2
import numpy as np

from implementation import ImageProcessing, OpenCVImageProcessing
from implementation.cat_image_processor import CatImageProcessor
from implementation.cat_image import AbstractCatImage, ColorCatImage, GrayscaleCatImage
from functools import wraps
import types


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result

    return wrapper


def validate_image_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Первый аргумент - self (экземпляр cat_image)
        self = args[0]
        if not hasattr(self, 'image_data') or self.image_data is None:
            raise ValueError("Image data is not loaded")
        if self.image_data.size == 0:
            raise ValueError("Image data is empty")
        return func(*args, **kwargs)

    return wrapper


def ensure_cat_images_output_dir():
    output_dir = "./cat_images_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_image_as_cat_image(image_path, image_type="color"):
    if image_type == "grayscale":
        return GrayscaleCatImage.from_file(image_path, f"file://{image_path}", "loaded", os.path.basename(image_path))
    else:
        return ColorCatImage.from_file(image_path, f"file://{image_path}", "loaded", os.path.basename(image_path))


def decorate_cat_image_methods(cat_image, mode):
    """Декорирует методы cat_image в зависимости от режима обработки"""

    # Список методов для декорирования для manual режима
    manual_methods = [
        'detect_edges_manual',
        'apply_convolution_manual',
        'convert_to_grayscale_manual'
    ]

    # Список методов для декорирования для opencv режима
    opencv_methods = [
        'detect_edges_opencv',
        'apply_convolution_opencv',
        'convert_to_grayscale_opencv'
    ]

    methods_to_decorate = manual_methods if mode == "manual" else opencv_methods

    for method_name in methods_to_decorate:
        if hasattr(cat_image, method_name):
            original_method = getattr(cat_image, method_name)

            def decorated_method(*args, **kwargs):
                if not hasattr(cat_image, 'image_data') or cat_image.image_data is None:
                    raise ValueError("Image data is not loaded")
                if cat_image.image_data.size == 0:
                    raise ValueError("Image data is empty")

                start_time = time.time()
                result = original_method(*args, **kwargs)
                execution_time = time.time() - start_time
                print(f"{method_name} executed in {execution_time:.4f} seconds")
                return result

            # Устанавливаем правильное имя для отладки
            decorated_method.__name__ = method_name

            # Заменяем метод
            setattr(cat_image, method_name, decorated_method)
            print(f"Decorated method: {method_name}")


def decorate_processor_methods(processor):
    methods_to_decorate = [
        'colorful_edge_detection',
        'gamma_correction',
        'corner_detection'
    ]

    for method_name in methods_to_decorate:
        if hasattr(processor, method_name):
            original_method = getattr(processor, method_name)
            decorated_method = log_execution_time(original_method)
            setattr(processor, method_name, decorated_method)
            print(f"Decorated processor method: {method_name}")


def process_cat_image_operations(cat_image1, cat_image2, operation, alpha=0.5):
    if cat_image1.image_data.shape != cat_image2.image_data.shape:
        raise ValueError(f"Image sizes don't match: {cat_image1.image_data.shape} vs {cat_image2.image_data.shape}")

    if operation == "add":
        return cat_image1 + cat_image2
    elif operation == "subtract":
        return cat_image1 - cat_image2
    elif operation == "blend":
        blended_data = cv2.addWeighted(cat_image1.image_data, alpha, cat_image2.image_data, 1 - alpha, 0)
        if isinstance(cat_image1, ColorCatImage):
            return ColorCatImage(blended_data, f"blended_{cat_image1.breed}", cat_image1.breed)
        else:
            return GrayscaleCatImage(blended_data, f"blended_{cat_image1.breed}", cat_image1.breed)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def main():
    parser = argparse.ArgumentParser(
        description="Image processing using ImageProcessing methods.",
    )

    parser.add_argument(
        "method",
        choices=[
            "blur", "sharpen", "sobel", "edges", "colorful-edges",
            "grayscale", "gamma-correction", "corners", "cat-api",
            "add", "subtract", "blend", "update"
        ],
        help="Processing method",
    )

    parser.add_argument(
        "input",
        help="Input image path or cat breed for cat-api",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output filename in cat_images_output folder",
    )

    parser.add_argument(
        "-g", "--gamma",
        type=float,
        default=1.5,
        help="Gamma value for correction",
    )

    parser.add_argument(
        "-c", "--convolution",
        default="blur",
        choices=["blur", "sharpen", "sobel"],
        help="Convolution type"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["manual", "opencv", "cat-api"],
        default="manual",
        help="Processing mode"
    )

    parser.add_argument(
        "-i2", "--input2",
        help="Second image for add/subtract/blend operations",
    )

    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.5,
        help="Blend coefficient for blend operation (0-1)"
    )

    parser.add_argument(
        "-b", "--brightness",
        type=float,
        default=1.2,
        help="Brightness coefficient for update operation"
    )

    parser.add_argument(
        "-ct", "--contrast",
        type=float,
        default=1.1,
        help="Contrast coefficient for update operation"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of images to download (cat-api only)"
    )

    parser.add_argument(
        "--breeds",
        nargs="+",
        help="List of breeds to process (cat-api only)"
    )

    parser.add_argument(
        "--image-type",
        choices=["color", "grayscale"],
        default="color",
        help="Image type for loading (local files only)"
    )

    args = parser.parse_args()

    if args.method == "cat-api" or args.mode == "cat-api":
        print("Using Cat API processor")
        processor = CatImageProcessor()

        image_type = getattr(args, 'image_type', 'color')

        if args.breeds:
            print(f"Processing breeds: {args.breeds} ({image_type} images)")
            processor.process_multiple_breeds(args.breeds, images_per_breed=args.limit, image_type=image_type)
        else:
            breed = args.input if args.input != "random" else None
            cat_images = processor.download_cat_images(limit=args.limit, breed=breed, image_type=image_type)
            if cat_images:
                processor.process_and_save_images(cat_images)
                print(f"Successfully processed {len(cat_images)} {image_type} images")
            else:
                print("Failed to download images")
        return

    try:
        cat_image = load_image_as_cat_image(args.input, args.image_type)
        image = cat_image.image_data
        print(f"Loaded {type(cat_image).__name__}: {args.input}")
        print(f"Image size: {cat_image.image_data.shape[1]}x{cat_image.image_data.shape[0]}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    cat_image2 = None
    image2 = None
    if args.input2:
        try:
            cat_image2 = load_image_as_cat_image(args.input2, args.image_type)
            image2 = cat_image2.image_data
            print(f"Loaded second image: {type(cat_image2).__name__}: {args.input2}")
            print(f"Second image size: {cat_image2.image_data.shape[1]}x{cat_image2.image_data.shape[0]}")
        except Exception as e:
            print(f"Error loading second image: {e}")
            return

    if args.mode == "manual":
        processor = ImageProcessing()
        print("Using manual implementation")
    else:
        processor = OpenCVImageProcessing()
        print("Using OpenCV implementation")

    print("Applying decorators to methods...")
    decorate_cat_image_methods(cat_image, args.mode)
    decorate_processor_methods(processor)

    start_time = time.time()

    if args.method in ["add", "subtract", "blend"]:
        if cat_image2 is None:
            print("Error: for this operation you need to specify second image via --input2")
            return

        try:
            result_cat_image = process_cat_image_operations(cat_image, cat_image2, args.method, args.alpha)
            result = result_cat_image.image_data

            if args.method == "add":
                print("Image addition completed")
            elif args.method == "subtract":
                print("Image subtraction completed")
            elif args.method == "blend":
                print(f"Image blending completed with alpha={args.alpha}")

        except ValueError as e:
            print(f"Error: {e}")
            print("Tip: use images of the same size")
            return
        except Exception as e:
            print(f"Error during operation: {e}")
            return

    else:
        if args.method == "edges":
            if args.mode == "manual":
                result = cat_image.detect_edges_manual()
            else:
                result = cat_image.detect_edges_opencv()

        elif args.method == "convolution":
            if args.mode == "manual":
                result = cat_image.apply_convolution_manual(args.convolution)
            else:
                result = cat_image.apply_convolution_opencv(args.convolution)

        elif args.method == "grayscale":
            if args.mode == "manual":
                result = cat_image.convert_to_grayscale_manual()
            else:
                result = cat_image.convert_to_grayscale_opencv()

        elif args.method == "colorful-edges":
            result = processor.colorful_edge_detection(image)

        elif args.method == "gamma-correction":
            result = processor.gamma_correction(image, args.gamma)

        elif args.method == "corners":
            result = processor.corner_detection(image)

        elif args.method == "update":
            result = image.astype(np.float32)
            result = result * args.contrast
            result = result + (args.brightness - 1) * 128
            result = np.clip(result, 0, 255).astype(np.uint8)
            print(f"Image updated: contrast={args.contrast}, brightness={args.brightness}")

        else:
            print("Error: unknown method")
            return

    execution_time = time.time() - start_time

    output_dir = ensure_cat_images_output_dir()

    if args.output:
        output_path = os.path.join(output_dir, args.output)
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        mode_suffix = "_opencv" if args.mode == "opencv" else "_manual"
        image_type_suffix = "_gray" if args.image_type == "grayscale" else "_color"

        if args.method in ["add", "subtract", "blend"] and args.input2:
            base2 = os.path.splitext(os.path.basename(args.input2))[0]
            operation_suffix = f"_{args.method}_{base2}"
        else:
            operation_suffix = f"_{args.method}"

        output_filename = f"{base_name}{operation_suffix}{image_type_suffix}{mode_suffix}.png"
        output_path = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")
    print(f"Total execution time: {execution_time:.4f} seconds")
    print(f"Processing mode: {args.mode}")
    print(f"Image type: {type(cat_image).__name__}")

    if args.method in ["add", "subtract", "blend"]:
        print(f"Result size: {result.shape[1]}x{result.shape[0]}")


if __name__ == "__main__":
    main()