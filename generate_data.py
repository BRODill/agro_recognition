import numpy as np
import random
import os

def generate_training_data(num_samples, num_features, num_classes):
    """Генерирует случайные обучающие данные."""
    features = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, num_classes, num_samples)
    return features, labels

def save_training_data(filename, features, labels):
    """Сохраняет обучающие данные в файл CSV."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Создаем каталог, если он не существует
    with open(filename, 'w') as f:
        for i in range(len(features)):
            row = ','.join(map(str, features[i])) + ',' + str(labels[i])
            f.write(row + '\n')

def generate_data():
    data = [random.randint(1, 100) for _ in range(10)]
    print("Generated data:", data)

def generate_training_data_method(method, out_path):
    """
    Генерирует фиктивную обучающую выборку для указанного метода.
    method: 'fft', 'wavelet', 'glcm'
    out_path: путь для сохранения csv
    """
    if method == 'fft':
        n_features = 6
    elif method == 'wavelet':
        n_features = 8
    elif method == 'glcm':
        n_features = 24
    else:
        raise ValueError("Неизвестный метод анализа")
    # 8 строк, 2 класса (0 и 1)
    data = []
    for i in range(8):
        features = np.round(np.linspace(0.1 + 0.1*i, 0.1 + 0.1*i + (n_features-1)*0.1, n_features), 2)
        label = 0 if i < 4 else 1
        row = list(features) + [label]
        data.append(row)
    np.savetxt(out_path, np.array(data), delimiter=",", fmt="%g")
    print(f"Сгенерирован файл: {out_path}")

if __name__ == "__main__":
    # Параметры для случайной генерации (опционально)
    num_samples = 100  # Количество образцов
    num_features = 4  # Количество признаков
    num_classes = 3  # Количество классов

    # Генерируем случайные данные (пример)
    features, labels = generate_training_data(num_samples, num_features, num_classes)
    filename = 'data/training_data/training_data_random.csv'
    save_training_data(filename, features, labels)
    print(f"Обучающие данные сгенерированы и сохранены в файл: {filename}")

    # Генерируем фиксированные выборки для каждого метода
    base = os.path.join(os.path.dirname(__file__), "data", "training_data")
    generate_training_data_method('fft', os.path.join(base, "training_data_fft.csv"))
    generate_training_data_method('wavelet', os.path.join(base, "training_data_wavelet.csv"))
    generate_training_data_method('glcm', os.path.join(base, "training_data.csv"))