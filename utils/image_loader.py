import cv2 # type: ignore

def load_image(image_path):
    """Загружает изображение из файла."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка: Не удалось прочитать изображение из {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None
