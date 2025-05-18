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