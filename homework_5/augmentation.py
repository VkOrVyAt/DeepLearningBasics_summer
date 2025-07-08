import os
import random
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import CustomImageDataset

# Установка путей
DATA_DIR = 'homework_5/data/train'
RESULTS_DIR = 'homework_5/result/task1'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Определение аугментаций
augmentations = {
    'Horizontal Flip': transforms.RandomHorizontalFlip(p=1.0),
    'Random Crop': transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    'Color Jitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    'Random Rotation': transforms.RandomRotation(degrees=30),
    'Grayscale': transforms.RandomGrayscale(p=1.0)
}

# Комбинированный пайплайн аугментаций
combined_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomGrayscale(p=0.3)
])

# Инициализация датасета без трансформаций
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Группировка изображений по классам
images_by_class = defaultdict(list)
for img_path, label in zip(dataset.images, dataset.labels):
    class_name = class_names[label]
    images_by_class[class_name].append(img_path)

# Выбор по одному случайному изображению из пяти разных классов
selected_images = []
selected_labels = []
for class_name in class_names[:5]:  # Берем первые 5 классов
    img_path = random.choice(images_by_class[class_name])
    selected_images.append(img_path)
    selected_labels.append(class_name)

# Функция визуализации
def visualize_augmentations(image_path, class_name, img_idx):
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Создание сетки 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Применение каждой аугментации отдельно
    for idx, (aug_name, aug) in enumerate(augmentations.items(), 1):
        aug_image = aug(image)
        axes[idx].imshow(aug_image)
        axes[idx].set_title(aug_name)
        axes[idx].axis('off')
    
    # Применение комбинированных аугментаций
    combined_image = combined_transform(image)
    axes[5].imshow(combined_image)
    axes[5].set_title('Combined')
    axes[5].axis('off')
    
    plt.suptitle(f'Class: {class_name}')
    plt.tight_layout()
    
    # Сохранение результата
    output_path = os.path.join(RESULTS_DIR, f'aug_{img_idx}_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

# Применение и визуализация для каждого изображения
for idx, (img_path, class_name) in enumerate(zip(selected_images, selected_labels)):
    visualize_augmentations(img_path, class_name, idx)

print(f"Результаты сохранены в {RESULTS_DIR}")