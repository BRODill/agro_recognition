from PIL import Image, ImageFilter

def preprocess_image(image_path):
    """Выполняет шумоподавление и коррекцию изображения."""
    image = Image.open(image_path)
    image = image.filter(ImageFilter.MedianFilter(size=3))  # Шумоподавление
    return image
