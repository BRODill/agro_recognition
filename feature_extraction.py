from core.fft_analysis import perform_fft

def extract_features(image, method):
    """Вычисляет признаки на основе выбранного метода."""
    if method == "fft":
        features = perform_fft(image)
        print(f"Признаки, извлечённые методом FFT: {features}")
    else:
        raise ValueError(f"Метод анализа {method} временно отключён. Используйте только 'fft'.")
