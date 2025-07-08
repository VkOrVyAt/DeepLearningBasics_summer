import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import CustomImageDataset
import torch
import numpy as np
from extra_augs import AddGaussianNoise, RandomErasingCustom, Solarize

# Установка путей
DATA_DIR = 'D:/ML/homework_5/data/train'
RESULTS_DIR = 'homework_5/result/task2'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Кастомные аугментации
class RandomPerspective:
    """Применяет случайное перспективное искажение."""
    def __init__(self, distortion_scale=0.3):
        self.distortion_scale = distortion_scale
    
    def __call__(self, img):
        # Получаем размеры изображения
        width, height = img.size
        # Определяем случайные смещения для углов
        startpoints = [(0, 0), (width, 0), (width, height), (0, height)]
        endpoints = [
            (random.uniform(0, self.distortion_scale * width), random.uniform(0, self.distortion_scale * height)),
            (width - random.uniform(0, self.distortion_scale * width), random.uniform(0, self.distortion_scale * height)),
            (width - random.uniform(0, self.distortion_scale * width), height - random.uniform(0, self.distortion_scale * height)),
            (random.uniform(0, self.distortion_scale * width), height - random.uniform(0, self.distortion_scale * height))
        ]
        # Применяем перспективное преобразование
        return img.transform(img.size, Image.PERSPECTIVE, self._compute_perspective_coeffs(startpoints, endpoints), Image.BICUBIC)
    
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
        # Случайно выбираем тип размытия
        filter_type = random.choice(self.filters)
        if filter_type == ImageFilter.GaussianBlur:
            return img.filter(filter_type(radius=self.radius))
        else:
            return img.filter(filter_type(size=max(3, int(self.radius * 2 + 1))))

class AdjustSharpness:
    """Регулирует резкость изображения."""
    def __init__(self, factor=2.0):
        self.factor = factor
    
    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(self.factor)

# Определение аугментаций
custom_augs = {
    'Perspective': RandomPerspective(distortion_scale=0.3),
    'Blur': RandomBlur(radius=2),
    'Sharpness': AdjustSharpness(factor=2.0)
}

extra_augs = {
    'Gaussian Noise': AddGaussianNoise(mean=0.0, std=0.1),
    'Random Erasing': RandomErasingCustom(p=1.0, scale=(0.02, 0.2)),
    'Solarize': Solarize(threshold=128)
}

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Выбор случайных изображений из первых 5 классов
selected_images = []
selected_labels = []
for class_name in class_names[:5]:
    class_idx = dataset.class_to_idx[class_name]
    class_images = [i for i, lbl in enumerate(dataset.labels) if lbl == class_idx]
    random_idx = random.choice(class_images)
    img_path = dataset.images[random_idx]
    selected_images.append(img_path)
    selected_labels.append(class_name)

# Функция для визуализации
def visualize_augmentations(img_path, class_name, img_idx):
    # Загрузка изображения
    image = Image.open(img_path).convert('RGB')
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    image_tensor = transforms.ToTensor()(image)
    
    # Создание сетки 2x4 (оригинал + 3 кастомные + 3 из extra_augs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Кастомные аугментации
    for idx, (aug_name, aug) in enumerate(custom_augs.items(), 1):
        aug_image = aug(image)
        axes[idx].imshow(aug_image)
        axes[idx].set_title(aug_name)
        axes[idx].axis('off')
    
    # Аугментации из extra_augs
    for idx, (aug_name, aug) in enumerate(extra_augs.items(), 4):
        aug_image = aug(image_tensor)
        axes[idx].imshow(aug_image.permute(1, 2, 0).numpy())
        axes[idx].set_title(aug_name)
        axes[idx].axis('off')
    
    # Пустой слот (для выравнивания сетки)
    axes[7].axis('off')
    
    plt.suptitle(f'Class: {class_name}')
    plt.tight_layout()
    
    # Сохранение результата
    output_path = os.path.join(RESULTS_DIR, f'custom_aug_{img_idx}_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

# Применение и визуализация
for idx, (img_path, class_name) in enumerate(zip(selected_images, selected_labels)):
    visualize_augmentations(img_path, class_name, idx)

print(f"Результаты сохранены в {RESULTS_DIR}")