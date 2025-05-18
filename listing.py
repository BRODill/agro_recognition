# Листинг программы AgroRecognition

# GUI интерфейс
# filepath: e:\agro\agro_recognition\agro_recognition\gui.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# -*- coding: utf-8 -*-
import tkinter as tk  # type: ignore
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # type: ignore
import subprocess
import os
import sys
import locale  # Для определения системной кодировки

class AgroRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Распознавание агроугодий")

        self.image_path = tk.StringVar()
        self.analysis_method = tk.StringVar(value="glcm")  # Значение по умолчанию

        # Элементы интерфейса
        self.label_image = tk.Label(master, text="Изображение:")
        self.label_image.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_image = tk.Entry(master, textvariable=self.image_path, width=50)
        self.entry_image.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.button_browse = tk.Button(master, text="Обзор", command=self.browse_image)
        self.button_browse.grid(row=0, column=2, padx=5, pady=5)

        self.label_method = tk.Label(master, text="Метод анализа:")
        self.label_method.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.radio_fft = tk.Radiobutton(master, text="FFT", variable=self.analysis_method, value="fft")
        self.radio_fft.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.radio_wavelet = tk.Radiobutton(master, text="Wavelet", variable=self.analysis_method, value="wavelet")
        self.radio_wavelet.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.radio_glcm = tk.Radiobutton(master, text="GLCM", variable=self.analysis_method, value="glcm")
        self.radio_glcm.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        self.button_run = tk.Button(master, text="Запуск", command=self.run_analysis)
        self.button_run.grid(row=3, column=1, columnspan=2, padx=5, pady=10)  # Переместили на строку ниже

        self.status_label = tk.Label(master, text="")
        self.status_label.grid(row=4, column=0, columnspan=4, padx=5, pady=5)

        # Добавление опции предварительной обработки
        self.preprocess_var = tk.BooleanVar(value=False)
        self.checkbox_preprocess = tk.Checkbutton(master, text="Предварительная обработка (шумоподавление)", variable=self.preprocess_var)
        self.checkbox_preprocess.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Кнопка для отображения результатов
        self.button_show_results = tk.Button(master, text="Показать результаты", command=self.show_results)
        self.button_show_results.grid(row=5, column=1, columnspan=2, padx=5, pady=10)  # Переместили ниже

        # Настройка сетки для растягивания entry
        master.columnconfigure(1, weight=1)

    def browse_image(self):
        """Открывает диалоговое окно выбора файла."""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Выберите изображение",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;"), ("all files", "*.*"))
        )
        if filename:
            self.image_path.set(filename)

    def run_analysis(self):
        """Запускает анализ с использованием выбранных параметров."""
        image_path = self.image_path.get()
        analysis_method = self.analysis_method.get()
        preprocess = self.preprocess_var.get()

        if not image_path:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите изображение.")
            return

        if not analysis_method:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите метод анализа.")
            return

        # Сформировать команду для запуска main.py
        command = [
            sys.executable,  # Используем текущий интерпретатор Python
            "main.py",
            "--method",
            analysis_method,
            image_path
        ]
        if preprocess:
            command.append("--preprocess")

        print(f"Запуск команды: {' '.join(command)}")  # Отладочный вывод команды

        try:
            self.status_label.config(text="Выполнение анализа...")
            self.master.update()  # Обновить интерфейс, чтобы отобразить сообщение

            # Принудительно устанавливаем кодировку UTF-8
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate()

            print("Результат выполнения команды:")  # Отладочный вывод
            print("STDOUT:")
            print(stdout)  # Вывод stdout
            print("STDERR:")
            print(stderr)  # Вывод stderr

            if process.returncode == 0:
                self.status_label.config(text="Анализ завершен успешно.")
                print("Вывод main.py:")
                print(stdout)  # Вывести результат в консоль
            else:
                self.status_label.config(text="Ошибка при выполнении анализа.")
                print("Ошибки main.py:")
                print(stderr)  # Вывести сообщение об ошибке в консоль

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
            self.status_label.config(text="Ошибка")

    def show_results(self):
        """Открывает окно с результатами анализа."""
        results_path = os.path.join(os.getcwd(), "data", "classification_map.png")
        print(f"Проверка наличия файла результатов: {results_path}")

        if os.path.exists(results_path):
            result_window = tk.Toplevel(self.master)
            result_window.title("Результаты анализа")

            try:
                img = Image.open(results_path)
                img = img.resize((500, 500), Image.ANTIALIAS)
                img_tk = ImageTk.PhotoImage(img)

                label_result = tk.Label(result_window, image=img_tk)
                label_result.image = img_tk
                label_result.pack()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл результатов: {e}")
        else:
            messagebox.showerror("Ошибка", "Результаты не найдены. Убедитесь, что анализ завершен.")

root = tk.Tk()
gui = AgroRecognitionGUI(root)
root.mainloop()

# Модуль обучения модели
# filepath: e:\agro\agro_recognition\agro_recognition\train_model.py
import os
from classification.svm_classifier import train_svm

# Путь к файлу с обучающими данными
current_dir = os.path.dirname(__file__)
training_data_path = os.path.join(current_dir, "data", "training_data", "training_data.csv")

print(f"Путь к файлу с обучающими данными: {training_data_path}")

# Проверяем, существует ли файл
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"Файл с обучающими данными не найден: {training_data_path}")

# Обучение модели
train_svm(features=None, training_data_path=training_data_path)

# Модуль анализа FFT
# filepath: e:\agro\agro_recognition\agro_recognition\core\fft_analysis.py
import numpy as np  # type: ignore
import cv2  # type: ignore
import sys
# Устанавливаем кодировку UTF-8 для вывода
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def perform_fft(image):
    """Выполняет преобразование Фурье и возвращает спектр мощности."""
    # Преобразуем в градации серого, если изображение цветное
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Преобразуем в 8-битное изображение (uint8), если это необходимо
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))  # Спектр мощности

    # Отладочный вывод полного спектра Фурье
    print("Полный спектр Фурье (массив):")
    print(magnitude_spectrum)

    # Извлечение признаков
    mean = np.mean(magnitude_spectrum)
    std = np.std(magnitude_spectrum)

    height, width = magnitude_spectrum.shape
    center_x, center_y = width // 2, height // 2

    energy_q1 = np.sum(magnitude_spectrum[:center_y, :center_x])
    energy_q2 = np.sum(magnitude_spectrum[:center_y, center_x:])
    energy_q3 = np.sum(magnitude_spectrum[center_y:, :center_x])
    energy_q4 = np.sum(magnitude_spectrum[center_y:, center_x:])

    features = [mean, std, energy_q1, energy_q2, energy_q3, energy_q4]
    return np.array(features)

# Модуль анализа Wavelet
# filepath: e:\agro\agro_recognition\agro_recognition\core\wavelet_analysis.py
import pywt
import numpy as np  # type: ignore
import cv2  # type: ignore
import sys
# Устанавливаем кодировку UTF-8 для вывода
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def perform_wavelet(image):
    """Выполняет вейвлет-анализ."""
    # Преобразуем в градации серого, если изображение цветное
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Выполняем дискретное вейвлет-преобразование
    coeffs = pywt.dwt2(gray, 'haar')  # 'haar' - простой вейвлет
    cA, (cH, cV, cD) = coeffs

    # Извлекаем признаки
    features = [
        np.mean(cA), np.std(cA),
        np.mean(cH), np.std(cH),
        np.mean(cV), np.std(cV),
        np.mean(cD), np.std(cD)
    ]
    return np.array(features)

# Модуль анализа GLCM
# filepath: e:\agro\agro_recognition\agro_recognition\core\glcm_analysis.py
import numpy as np  # type: ignore
from skimage.feature import graycomatrix, graycoprops  # type: ignore
import cv2  # type: ignore
import sys
# Устанавливаем кодировку UTF-8 для вывода
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def calculate_glcm_features(image, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Вычисляет текстурные характеристики GLCM."""
    # Преобразуем в градации серого, если изображение цветное
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Вычисляем GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)

    # Извлекаем текстурные характеристики
    features = np.concatenate([
        graycoprops(glcm, 'contrast').flatten(),
        graycoprops(glcm, 'dissimilarity').flatten(),
        graycoprops(glcm, 'homogeneity').flatten(),
        graycoprops(glcm, 'energy').flatten(),
        graycoprops(glcm, 'correlation').flatten(),
        graycoprops(glcm, 'ASM').flatten()
    ])
    return features

# Модуль классификации SVM
# filepath: e:\agro\agro_recognition\agro_recognition\classification\svm_classifier.py
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
import os
import sys
# Устанавливаем кодировку UTF-8 для вывода
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def train_svm(features, training_data_path):
    """Обучает SVM классификатор."""
    try:
        data = np.loadtxt(training_data_path, delimiter=',')
        print(f"Загруженные данные из {training_data_path}:")
        print(data)  # Отладочный вывод данных
        X = data[:, :-1]  # Все столбцы, кроме последнего (признаки)
        y = data[:, -1]  # Последний столбец (метки классов)
    except Exception as e:
        print(f"Ошибка при загрузке обучающих данных: {e}")
        return None

    # Проверяем распределение классов
    unique, counts = np.unique(y, return_counts=True)
    print(f"Распределение классов: {dict(zip(unique, counts))}")
    if len(unique) < 2:
        print("Предупреждение: Недостаточно классов для обучения модели.")
    if any(count < 2 for count in counts):
        print("Предупреждение: Некоторые классы имеют слишком мало примеров.")

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создаем SVM классификатор
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)  # Используем RBF ядро и изменяем параметр C

    # Обучаем модель
    model.fit(X_train, y_train)

    # Оцениваем точность на тестовой выборке
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность SVM классификатора на тестовой выборке: {accuracy:.2f}")

    # Сохраняем обученную модель в файл (чтобы не обучать каждый раз)
    model_filename = os.path.join("data", "svm_model.joblib")  # Сохраняем в папку data
    joblib.dump(model, model_filename)
    print(f"Обученная модель сохранена в файл: {model_filename}")

    return model


def predict_svm(features, model):
    """Предсказывает классы на основе обученной SVM модели."""
    if features.ndim == 1:
        features = features.reshape(1, -1)
    predictions = model.predict(features)
    print(f"Предсказания SVM: {predictions}")
    return predictions

# Основной модуль классификации
# filepath: e:\agro\agro_recognition\agro_recognition\classification.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from classification.svm_classifier import predict_svm
import joblib
import os
import numpy as np

def classify_image(features):
    """Классифицирует изображение на основе признаков."""
    model_path = os.path.join("data", "svm_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель SVM не найдена по пути: {model_path}. Пожалуйста, обучите модель.")

    # Загружаем обученную модель
    print(f"Загрузка модели SVM из {model_path}...")  # Отладочный вывод
    try:
        model = joblib.load(model_path)
        print("Модель SVM успешно загружена.")  # Отладочный вывод
    except Exception as e:
        print(f"Ошибка при загрузке модели SVM: {e}")
        raise ValueError("Не удалось загрузить модель SVM.")

    # Выполняем предсказание
    print("Выполняется предсказание модели SVM...")  # Отладочный вывод
    try:
        predictions = predict_svm(features, model)
        print(f"Предсказания модели: {predictions}")  # Отладочный вывод предсказаний
    except Exception as e:
        print(f"Ошибка при выполнении предсказания: {e}")
        raise ValueError("Ошибка при выполнении предсказания модели SVM.")

    # Проверяем, что предсказания корректны
    if predictions.size == 0:
        raise ValueError("Предсказания модели пусты. Проверьте входные данные и модель.")

    # Преобразуем предсказания в числовую карту классификации
    try:
        classification_map = np.array(predictions, dtype=int)
        print(f"Карта классификации: {classification_map}")  # Отладочный вывод карты классификации
    except Exception as e:
        print(f"Ошибка при преобразовании предсказаний в карту классификации: {e}")
        raise ValueError("Ошибка при преобразовании предсказаний в карту классификации.")

    return classification_map

# Модуль извлечения признаков
# filepath: e:\agro\agro_recognition\agro_recognition\feature_extraction.py
from core.fft_analysis import perform_fft

def extract_features(image, method):
    """Вычисляет признаки на основе выбранного метода."""
    if method == "fft":
        features = perform_fft(image)
        print(f"Признаки, извлечённые методом FFT: {features}")
    else:
        raise ValueError(f"Метод анализа {method} временно отключён. Используйте только 'fft'.")

# Основной модуль программы
# filepath: e:\agro\agro_recognition\agro_recognition\main.py
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
    output_dir = os.path.join(os.getcwd(), "data")  # Сохраняем в папку data
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

# Файл зависимостей
# filepath: e:\agro\agro_recognition\agro_recognition\requirements.txt
numpy
scipy
opencv-python
scikit-image
scikit-learn
matplotlib
pillow
PyWavelets
tk
