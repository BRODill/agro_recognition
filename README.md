# AgroRecognition

## Описание

AgroRecognition — это программа для автоматического анализа и классификации агроугодий по изображениям с использованием методов обработки изображений и машинного обучения (SVM). Поддерживаются методы анализа: FFT (преобразование Фурье), Wavelet (вейвлет-анализ), GLCM (признаки на основе матрицы совместной встречаемости градаций серого).

## Структура проекта

```
agro_recognition/
│
├── core/
│   ├── fft_analysis.py
│   ├── wavelet_analysis.py
│   └── glcm_analysis.py
├── classification/
│   ├── svm_classifier.py
│   ├── classification.py
│   └── __init__.py
├── preprocessing/
│   ├── noise_reduction.py
│   └── image_preprocessor.py
├── utils/
│   └── image_loader.py
├── data/
│   ├── training_data/
│   │   └── training_data.csv
│   ├── images/
│   │   └── test.jpg
│   └── classification_map.png
├── feature_extraction.py
├── train_model.py
├── main.py
├── gui.py
├── requirements.txt
└── README.md
```

## Установка зависимостей

1. Создайте виртуальное окружение (рекомендуется):

   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. Установите зависимости:

   ```
   pip install -r requirements.txt
   ```

## Использование

### Обучение модели

Перед использованием классификации обучите модель SVM на ваших данных:

```
python train_model.py
```

### Запуск анализа через консоль

```
python main.py --method fft data/images/test.jpg
python main.py --method wavelet data/images/test.jpg
python main.py --method glcm data/images/test.jpg
```

Параметры:
- `--method` — метод анализа (`fft`, `wavelet`, `glcm`)
- `--preprocess` — включить предварительную обработку (шумоподавление)
- `--output` — путь для сохранения карты классификации (по умолчанию `data/classification_map.png`)
- `image_path` — путь к изображению для анализа

### Запуск графического интерфейса

```
python gui.py
```

В графическом интерфейсе выберите изображение, метод анализа и запустите обработку.

## Описание модулей

- **core/fft_analysis.py** — извлечение признаков с помощью преобразования Фурье.
- **core/wavelet_analysis.py** — извлечение признаков с помощью вейвлет-анализа.
- **core/glcm_analysis.py** — извлечение признаков на основе матрицы совместной встречаемости градаций серого (GLCM).
- **classification/svm_classifier.py** — обучение и предсказание SVM.
- **classification/classification.py** — функция классификации изображения.
- **feature_extraction.py** — выбор метода извлечения признаков.
- **main.py** — основной модуль для запуска анализа.
- **gui.py** — графический интерфейс пользователя.
- **train_model.py** — обучение модели SVM.
- **requirements.txt** — список зависимостей.

## Требования

- Python 3.8+
- numpy, scipy, opencv-python, scikit-image, scikit-learn, matplotlib, pillow, PyWavelets, tk

## Пример запуска

```
python main.py --method fft data/images/test.jpg --preprocess
```

## Автор

BRODill (Диль Максим), 2025

# Чтобы исправить ошибку git "Author identity unknown", выполните в командной строке:

git config --global user.name "BRODill"
git config --global user.email "your_email@example.com"

# Замените "your_email@example.com" на ваш реальный email, связанный с GitHub.
# После этого повторите команду git commit.

