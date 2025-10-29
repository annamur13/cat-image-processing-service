"""
main.py
Автор: [Мурзина Анна 6405-010302D]
"""

import argparse
import os
import time
import cv2
import numpy as np

from implementation import ImageProcessing, OpenCVImageProcessing
from implementation.cat_image_processor import CatImageProcessor
from implementation.cat_image import CatImage

def resize_images_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Используем меньший размер из двух изображений
    new_width = min(w1, w2)
    new_height = min(h1, h2)

    img1_resized = cv2.resize(img1, (new_width, new_height))
    img2_resized = cv2.resize(img2, (new_width, new_height))

    print(f"Изображения приведены к размеру: {new_width}x{new_height}")
    return img1_resized, img2_resized

def ensure_cat_images_output_dir() -> str:
    output_dir = "./cat_images_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing.",
    )

    parser.add_argument(
        "method",
        choices=[
            "blur", "sharpen", "sobel", "edges", "colorful-edges",
            "grayscale", "gamma-correction", "corners", "cat-api",
            "add", "subtract", "blend", "update"
        ],
        help="Метод обработки",
    )

    parser.add_argument(
        "input",
        help="Путь к входному изображению или порода кота для cat-api",
    )

    parser.add_argument(
        "-o", "--output",
        help="Имя файла для сохранения в папке cat_images_output",
    )

    parser.add_argument(
        "-g", "--gamma",
        type=float,
        default=1.5,
        help="Значение gamma для коррекции",
    )

    parser.add_argument(
        "-c", "--convolution",
        default="blur",
        choices=["blur", "sharpen", "sobel"],
        help="Тип свертки"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["manual", "opencv", "cat-api"],
        default="manual",
        help="Режим обработки"
    )

    parser.add_argument(
        "-i2", "--input2",
        help="Второе изображение для операций сложения/вычитания/смешивания",
    )

    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.5,
        help="Коэффициент смешивания для blend операции (0-1)"
    )

    parser.add_argument(
        "-b", "--brightness",
        type=float,
        default=1.2,
        help="Коэффициент яркости для update операции"
    )

    parser.add_argument(
        "-ct", "--contrast",
        type=float,
        default=1.1,
        help="Коэффициент контрастности для update операции"
    )

    # Новые аргументы для cat-api
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Количество изображений для скачивания (только для cat-api)"
    )

    parser.add_argument(
        "--breeds",
        nargs="+",
        help="Список пород для обработки (только для cat-api)"
    )

    args = parser.parse_args()

    if args.mode == "cat-api":
        print("🐱 Используется Cat API процессор")
        processor = CatImageProcessor()
        if args.breeds:
            print(f"Обработка пород: {args.breeds}")
            processor.process_multiple_breeds(args.breeds, images_per_breed=args.limit)
        else:
            breed = args.input if args.input != "random" else None
            cat_images = processor.download_cat_images(limit=args.limit, breed=breed)
            if cat_images:
                processor.process_and_save_images(cat_images)
                print(f"Успешно обработано {len(cat_images)} изображений")
            else:
                print("Не удалось скачать изображения")
        return

    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    image2 = None
    if args.input2:
        image2 = cv2.imread(args.input2)
        if image2 is None:
            print(f"Ошибка: не удалось загрузить второе изображение {args.input2}")
            return

    if args.mode == "manual":
        processor = ImageProcessing()
        print("Используется ручная реализация")
    else:
        processor = OpenCVImageProcessing()
        print("Используется OpenCV реализация")

    start_time = time.time()

    if args.method == "edges":
        result = processor.edge_detection(image)

    elif args.method == "convolution":
        if args.mode == "manual":
            result = processor.convolution(image, args.convolution)
        else:
            if args.convolution == "blur":
                result = processor.gaussian_blur(image)
            elif args.convolution == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                result = processor.convolution(image, kernel)
            elif args.convolution == "sobel":
                result = processor.sobel_edge_detection(image)

    elif args.method == "grayscale":
        result = processor.rgb_to_grayscale(image)

    elif args.method == "colorful-edges":
        result = processor.colorful_edge_detection(image)

    elif args.method == "gamma-correction":
        result = processor.gamma_correction(image, args.gamma)

    elif args.method == "corners":
        result = processor.corner_detection(image)

    elif args.method == "add":
        if image2 is None:
            print("Ошибка: для сложения нужно указать второе изображение через --input2")
            return
        image_resized, image2_resized = resize_images_to_match(image, image2)
        result = cv2.add(image_resized, image2_resized)
        print("Выполнено сложение изображений")

    elif args.method == "subtract":
        if image2 is None:
            print("Ошибка: для вычитания нужно указать второе изображение через --input2")
            return
        image_resized, image2_resized = resize_images_to_match(image, image2)
        result = cv2.subtract(image_resized, image2_resized)
        print("Выполнено вычитание изображений")

    elif args.method == "blend":
        if image2 is None:
            print("Ошибка: для смешивания нужно указать второе изображение через --input2")
            return
        image_resized, image2_resized = resize_images_to_match(image, image2)
        result = cv2.addWeighted(image_resized, args.alpha, image2_resized, 1 - args.alpha, 0)
        print(f"Выполнено смешивание изображений с alpha={args.alpha}")

    elif args.method == "update":
        result = image.astype(np.float32)
        result = result * args.contrast  # Контраст
        result = result + (args.brightness - 1) * 128  # Яркость
        result = np.clip(result, 0, 255).astype(np.uint8)
        print(f"Обновлено изображение: контраст={args.contrast}, яркость={args.brightness}")

    else:
        print("Ошибка: неизвестный метод")
        return

    execution_time = time.time() - start_time

    output_dir = ensure_cat_images_output_dir()

    if args.output:
        output_path = os.path.join(output_dir, args.output)
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        mode_suffix = "_opencv" if args.mode == "opencv" else "_manual"

        if args.method in ["add", "subtract", "blend"] and args.input2:
            base2 = os.path.splitext(os.path.basename(args.input2))[0]
            operation_suffix = f"_{args.method}_{base2}"
        else:
            operation_suffix = f"_{args.method}"

        output_filename = f"{base_name}{operation_suffix}{mode_suffix}.png"
        output_path = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в: {output_path}")
    print(f"Время выполнения: {execution_time:.4f} секунд")
    print(f"Режим обработки: {args.mode}")

    # Дополнительная информация для операций
    if args.method in ["add", "subtract", "blend"]:
        print(f"Размер результата: {result.shape[1]}x{result.shape[0]}")


if __name__ == "__main__":
    main()