import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import argparse
from utils.image_loader import load_image
from preprocessing.noise_reduction import reduce_noise
from core.fft_analysis import perform_fft
from core.wavelet_analysis import perform_wavelet
from core.glcm_analysis import calculate_glcm_features
from classification.svm_classifier import train_svm, predict_svm
import cv2  # type: ignore
import numpy as np  # type: ignore
import joblib
from preprocessing.image_preprocessor import preprocess_image  # Измените импорт, если проблема с __init__.py
from feature_extraction import extract_features
from classification import classify_image
import matplotlib.pyplot as plt  # Для визуализации и сохранения карты классификации


def save_classification_map(classification_map, output_path):
    """
    Сохраняет карту классификации в виде изображения.

    :param classification_map: numpy.ndarray, карта классификации
    :param output_path: str, путь для сохранения изображения
    """
    if not isinstance(classification_map, np.ndarray):
        raise ValueError("Карта классификации должна быть массивом numpy.")

    plt.imshow(classification_map, cmap="tab10")  # Используем цветовую карту для визуализации классов
    plt.colorbar(label="Классы агроугодий")
    plt.title("Карта классификации агроугодий")
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Анализ агроугодий")
    parser.add_argument("--method", type=str, choices=["fft", "wavelet", "glcm"], required=True, help="Метод анализа")
    parser.add_argument("--preprocess", action="store_true", help="Включить предварительную обработку")
    parser.add_argument("--output", type=str, default="classification_map.png", help="Путь для сохранения карты классификации")
    parser.add_argument("--training_data", type=str, help="Путь к файлу с обучающими данными для обучения модели")
    parser.add_argument("image_path", type=str, help="Путь к изображению")
    
    try:
        args = parser.parse_args()
        print(f"Полученные аргументы: {args}")  # Отладочный вывод аргументов
    except SystemExit as e:
        if e.code == 2:  # Ошибка аргументов
            print("Ошибка: Проверьте правильность переданных аргументов.")
            parser.print_help()
        sys.exit(1)

    # Проверяем существование файла изображения
    if not os.path.exists(args.image_path):
        print(f"Ошибка: Файл изображения не найден: {args.image_path}")
        sys.exit(1)

    # Устанавливаем путь для сохранения карты классификации
    output_dir = os.path.join(os.getcwd(), "data", "results")  # Сохраняем в папку data/results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Создаём папку, если её нет
    args.output = os.path.join(output_dir, "classification_map.png")
    print(f"Путь для сохранения карты классификации: {args.output}")  # Отладочный вывод

    # Предварительная обработка
    if args.preprocess:
        print("Выполняется предварительная обработка...")
        image = preprocess_image(args.image_path)
        if image is None:
            print("Ошибка: Не удалось выполнить предварительную обработку изображения.")
            sys.exit(1)
        print(f"Предварительная обработка завершена. Размер изображения: {image.shape if isinstance(image, np.ndarray) else 'Неизвестно'}")
    else:
        image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Ошибка: Не удалось загрузить изображение: {args.image_path}")
            sys.exit(1)

    # Преобразуем изображение в формат uint8, если это необходимо
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Вычисление признаков
    print("Вычисление признаков...")
    try:
        features = extract_features(image, method=args.method)
        if features is None:
            raise ValueError("Признаки не были вычислены.")
    except Exception as e:
        print(f"Ошибка при вычислении признаков: {str(e)}")
        sys.exit(1)
    print(f"Признаки вычислены: {features}")

    # Классификация
    print("Начало классификации...")  # Отладочный вывод
    try:
        classification_map = classify_image(features)
        if not isinstance(classification_map, np.ndarray):
            raise ValueError("Карта классификации должна быть массивом numpy.")
    except Exception as e:
        print(f"Ошибка при классификации: {str(e).encode('utf-8').decode('utf-8')}")
        sys.exit(1)
    print(f"Классификация выполнена. Карта классификации: {classification_map}")  # Отладочный вывод

    # Сохранение карты классификации
    print(f"Сохранение карты классификации в {args.output}...")
    try:
        print(f"Тип данных карты классификации: {type(classification_map)}")  # Отладочный вывод
        print(f"Содержимое карты классификации: {classification_map}")  # Отладочный вывод
        save_classification_map(classification_map, args.output)
    except Exception as e:
        print(f"Ошибка при сохранении карты классификации: {str(e).encode('utf-8').decode('utf-8')}")
        sys.exit(1)
    print("Карта классификации успешно сохранена.")

if __name__ == "__main__":
    main()