"""
main.py
Автор: [Мурзина Анна 6405-010302D]
"""

import argparse
import os
import time

import cv2
import numpy as np

from implementation import ImageProcessing
from implementation import OpenCVImageProcessing

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing.",
    )
    parser.add_argument(
        "method",
        choices=[
            "blur",
            "sharpen",
            "sobel",
            "edges",
            "colorful-edges",
            "grayscale",
            "gamma-correction",
            "corners",
        ],
        help="Метод обработки: edges, corners, grayscale, gamma-correction, convolution",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )
    parser.add_argument(
        "-g", "--gamma",
        type=float,
        default=1.5,
        help="Значение gamma для коррекции (по умолчанию: 1.5, только для gamma-correction)",
    )
    parser.add_argument(
        "-c", "--convolution",
        default="blur",
        choices=["blur", "sharpen", "sobel"],
        help="Тип свертки: blur, sharpen, sobel"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["manual", "opencv"],
        default="manual",
        help="Режим обработки: manual (ручная реализация) или opencv (библиотека OpenCV)"
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    # Выбор процессора в зависимости от режима
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
    else:
        print("Ошибка: неизвестный метод")
        return

    execution_time = time.time() - start_time

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        mode_suffix = "_opencv" if args.mode == "opencv" else "_manual"
        output_path = f"{base}_{args.method}{mode_suffix}.png"

    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")
    print(f"Время выполнения: {execution_time:.4f} секунд")
    print(f"Режим обработки: {args.mode}")


if __name__ == "__main__":
    main()