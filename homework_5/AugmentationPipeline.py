import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, RandomErasingCustom, Solarize
from custom_aug_v2 import RandomPerspective, RandomBlur, AdjustSharpness

# Пути к данным и результатам
DATA_DIR = 'D:/ML/homework_5/data/train'  # Папка с тренировочными данными
RESULTS_DIR = 'D:/ML/homework_5/result/task4'  # Папка для результатов Задания 4
os.makedirs(RESULTS_DIR, exist_ok=True)  # Создаем папку, если не существует

class AugmentationPipeline:
    """Класс для управления пайплайном аугментаций."""
    
    def __init__(self):
        self.augmentations = {}  # Словарь для хранения аугментаций
    
    def add_augmentation(self, name, aug):
        """Добавляет аугментацию по имени."""
        self.augmentations[name] = aug
    
    def remove_augmentation(self, name):
        """Удаляет аугментацию по имени."""
        if name in self.augmentations:
            del self.augmentations[name]
    
    def apply(self, image):
        """Применяет все аугментации к изображению."""
        img = image
        for name, aug in self.augmentations.items():
            # Конвертация PIL в тензор для тензорных аугментаций
            if isinstance(img, Image.Image) and name in ['Solarize', 'GaussianNoise', 'RandomErasing']:
                img = transforms.ToTensor()(img)
                img = aug(img)
                img = torch.clamp(img, 0, 1)  # Нормализуем значения
            # Конвертация тензора в PIL для PIL-аугментаций
            elif isinstance(img, torch.Tensor) and name in ['Perspective', 'Blur', 'Sharpness', 'Rotation']:
                img = transforms.ToPILImage()(img)
                img = aug(img)
            else:
                img = aug(img)
        # Возвращаем PIL изображение
        if isinstance(img, torch.Tensor):
            img = torch.clamp(img, 0, 1)  # Финальная нормализация
            img = transforms.ToPILImage()(img)
        return img
    
    def get_augmentations(self):
        """Возвращает список имен аугментаций."""
        return list(self.augmentations.keys())

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None, target_size=(224, 224))  # Датасет без трансформаций
class_names = dataset.get_class_names()  # Список имен классов

# Выбор случайного изображения из первых 5 классов
selected_images = []
selected_labels = []
for class_name in class_names[:5]:
    class_idx = dataset.class_to_idx[class_name]  # Индекс класса
    class_images = [i for i, lbl in enumerate(dataset.labels) if lbl == class_idx]  # Индексы изображений
    random_idx = random.choice(class_images)  # Случайный выбор
    img_path = dataset.images[random_idx]  # Путь к изображению
    selected_images.append(img_path)
    selected_labels.append(class_name)

# Определение пайплайнов
light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation('Rotation', transforms.RandomRotation(degrees=20))  # Легкий поворот
light_pipeline.add_augmentation('Blur', RandomBlur(radius=1.5))  # Легкое размытие

medium_pipeline = AugmentationPipeline()
medium_pipeline.add_augmentation('RandomErasing', RandomErasingCustom(p=0.6, scale=(0.05, 0.2)))  # Затирание
medium_pipeline.add_augmentation('Sharpness', AdjustSharpness(factor=1.3))  # Умеренная резкость
medium_pipeline.add_augmentation('Perspective', RandomPerspective(distortion_scale=0.2))  # Умеренная перспектива

heavy_pipeline = AugmentationPipeline()
heavy_pipeline.add_augmentation('Solarize', Solarize(threshold=120))  # Соляризация
heavy_pipeline.add_augmentation('GaussianNoise', AddGaussianNoise(mean=0.0, std=0.15))  # Шум
heavy_pipeline.add_augmentation('Blur', RandomBlur(radius=3.0))  # Сильное размытие

# Функция визуализации
def visualize_and_save(img_path, pipeline, class_name, config_name, idx):
    """Применяет пайплайн и сохраняет сравнение."""
    image = Image.open(img_path).convert('RGB')  # Загружаем изображение
    image = image.resize((224, 224), Image.Resampling.LANCZOS)  # Ресайз до 224x224
    aug_image = pipeline.apply(image)  # Применяем аугментации
    
    # Сетка 1x2 для сравнения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)  # Оригинал
    ax1.set_title('Оригинал')
    ax1.axis('off')
    ax2.imshow(aug_image)  # Аугментированное
    ax2.set_title(f'Конфигурация: {config_name}')
    ax2.axis('off')
    plt.suptitle(f'Класс: {class_name} | Аугментации: {", ".join(pipeline.get_augmentations())}')  # Заголовок
    plt.tight_layout()
    
    # Сохранение результата
    output_path = os.path.join(RESULTS_DIR, f'{config_name}_class_{class_name}_{idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Применение и визуализация
pipelines = {
    'light': light_pipeline,
    'medium': medium_pipeline,
    'heavy': heavy_pipeline
}
for config_name, pipeline in pipelines.items():
    for idx, (img_path, class_name) in enumerate(zip(selected_images, selected_labels)):
        visualize_and_save(img_path, pipeline, class_name, config_name, idx)

print(f"Результаты сохранены в {RESULTS_DIR}")