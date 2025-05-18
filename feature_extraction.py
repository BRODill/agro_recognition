from core.fft_analysis import perform_fft
from core.wavelet_analysis import perform_wavelet
from core.glcm_analysis import calculate_glcm_features

def extract_features(image, method):
    """Вычисляет признаки на основе выбранного метода."""
    if method == "fft":
        features = perform_fft(image)
        print(f"Признаки, извлечённые методом FFT: {features}")
    elif method == "wavelet":
        features = perform_wavelet(image)
        print(f"Признаки, извлечённые методом Wavelet: {features}")
    elif method == "glcm":
        features = calculate_glcm_features(image)
        print(f"Признаки, извлечённые методом GLCM: {features}")
    else:
        raise ValueError(f"Метод анализа {method} не поддерживается. Используйте 'fft', 'wavelet' или 'glcm'.")
    return features
