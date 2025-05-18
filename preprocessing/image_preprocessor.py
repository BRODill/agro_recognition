import cv2  # type: ignore

def preprocess_image(image_path: str):
    """
    Выполняет предварительную обработку изображения.
    :param image_path: Путь к изображению.
    :return: Предварительно обработанное изображение.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Изображение по пути {image_path} не найдено.")
    # Пример обработки: изменение размера и нормализация
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image
