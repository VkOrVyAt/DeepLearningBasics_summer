import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

class RandomPerspective:
    """Применяет случайное перспективное искажение."""
    def __init__(self, distortion_scale=0.3):
        self.distortion_scale = distortion_scale
    
    def __call__(self, img):
        width, height = img.size  # Размеры изображения
        startpoints = [(0, 0), (width, 0), (width, height), (0, height)]  # Начальные точки
        endpoints = [
            (random.uniform(0, self.distortion_scale * width), random.uniform(0, self.distortion_scale * height)),
            (width - random.uniform(0, self.distortion_scale * width), random.uniform(0, self.distortion_scale * height)),
            (width - random.uniform(0, self.distortion_scale * width), height - random.uniform(0, self.distortion_scale * height)),
            (random.uniform(0, self.distortion_scale * width), height - random.uniform(0, self.distortion_scale * height))
        ]  # Конечные точки
        return img.transform(img.size, Image.PERSPECTIVE, self._compute_perspective_coeffs(startpoints, endpoints), Image.BICUBIC)  # Применяем перспективу
    
    def _compute_perspective_coeffs(self, startpoints, endpoints):
        """Вычисляет коэффициенты перспективного преобразования."""
        matrix = []
        for p1, p2 in zip(startpoints, endpoints):
            matrix.extend([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.extend([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.array(matrix).reshape(8, 8)
        B = np.array([pt for pt in endpoints]).reshape(8)
        coeffs = np.linalg.solve(A, B)
        return coeffs.tolist()

class RandomBlur:
    """Применяет случайное размытие (гауссово или медианное)."""
    def __init__(self, radius=2):
        self.radius = radius
        self.filters = [ImageFilter.GaussianBlur, ImageFilter.MedianFilter]
    
    def __call__(self, img):
        filter_type = random.choice(self.filters)  # Выбираем тип размытия
        if filter_type == ImageFilter.GaussianBlur:
            return img.filter(filter_type(radius=self.radius))  # Гауссово размытие
        else:
            size = int(self.radius * 2)  # Базовый размер фильтра
            size = max(3, size if size % 2 == 1 else size + 1)  # Гарантируем нечетный size >= 3
            return img.filter(filter_type(size=size))  # Медианное размытие

class AdjustSharpness:
    """Регулирует резкость изображения."""
    def __init__(self, factor=2.0):
        self.factor = factor
    
    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)  # Создаем объект для резкости
        return enhancer.enhance(self.factor)  # Применяем резкость