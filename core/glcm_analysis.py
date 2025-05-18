from skimage.feature import graycomatrix, graycoprops  # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
import sys
# Устанавливаем кодировку UTF-8 для вывода с игнорированием ошибок
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
def calculate_glcm_features(image, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Вычисляет текстурные характеристики GLCM для изображения (цветного или ч/б)."""
    # Преобразование в uint8 и обработка цветных изображений
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    features = np.concatenate([
        graycoprops(glcm, 'contrast').flatten(),
        graycoprops(glcm, 'dissimilarity').flatten(),
        graycoprops(glcm, 'homogeneity').flatten(),
        graycoprops(glcm, 'energy').flatten(),
        graycoprops(glcm, 'correlation').flatten(),
        graycoprops(glcm, 'ASM').flatten()
    ])
    return features