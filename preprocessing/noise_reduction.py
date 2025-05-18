import cv2 # type: ignore

def reduce_noise(image):
    """Уменьшает шум на изображении с использованием медианного фильтра."""
    # Медианный фильтр хорошо удаляет "соль и перец" шум
    denoised_image = cv2.medianBlur(image, 5)  # Размер ядра фильтра = 5
    return denoised_image
