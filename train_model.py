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
