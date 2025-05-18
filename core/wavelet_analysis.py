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