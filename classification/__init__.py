from .svm_classifier import predict_svm
import joblib
import os
import numpy as np

def classify_image(features):
    """Классифицирует изображение на основе признаков."""
    model_path = os.path.join("data", "svm_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель SVM не найдена по пути: {model_path}. Пожалуйста, обучите модель.")
    print(f"Загрузка модели SVM из {model_path}...")
    try:
        model = joblib.load(model_path)
        print("Модель SVM успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели SVM: {e}")
        raise ValueError("Не удалось загрузить модель SVM.")
    print("Выполняется предсказание модели SVM...")
    try:
        predictions = predict_svm(features, model)
        print(f"Предсказания модели: {predictions}")
    except Exception as e:
        print(f"Ошибка при выполнении предсказания: {e}")
        raise ValueError("Ошибка при выполнении предсказания модели SVM.")
    if predictions.size == 0:
        raise ValueError("Предсказания модели пусты. Проверьте входные данные и модель.")
    try:
        classification_map = np.array(predictions, dtype=int)
        print(f"Карта классификации: {classification_map}")
        # Приведение к двумерному виду для совместимости с plt.imshow
        if classification_map.ndim == 1:
            classification_map = classification_map.reshape(1, -1)
    except Exception as e:
        print(f"Ошибка при преобразовании предсказаний в карту классификации: {e}")
        raise ValueError("Ошибка при преобразовании предсказаний в карту классификации.")
    return classification_map
