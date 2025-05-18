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

if __name__ == "__main__":
    # Параметры
    num_samples = 100  # Количество образцов
    num_features = 4  # Количество признаков
    num_classes = 3  # Количество классов

    # Генерируем данные
    features, labels = generate_training_data(num_samples, num_features, num_classes)

    # Сохраняем данные в файл
    filename = 'data/training_data/training_data.csv'
    save_training_data(filename, features, labels)

    print(f"Обучающие данные сгенерированы и сохранены в файл: {filename}")

    generate_data()