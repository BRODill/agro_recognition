import numpy as np  # type: ignore
import cv2  # type: ignore
import sys
# Устанавливаем кодировку UTF-8 для вывода
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def perform_fft(image):
    """Выполняет преобразование Фурье и возвращает спектр мощности."""
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