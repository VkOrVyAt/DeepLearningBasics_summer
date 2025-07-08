import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import CustomImageDataset
from PIL import Image
import numpy as np
import pandas as pd

# Set paths
DATA_DIR = 'homework_5/data/train'
RESULTS_DIR = 'homework_5/result/task3'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize dataset
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None)

# Count images per class
class_counts = {}
for class_name in dataset.classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    class_counts[class_name] = len(os.listdir(class_dir))

# Collect image sizes
sizes = []
for img_path in dataset.images:
    with Image.open(img_path) as img:
        sizes.append(img.size)

# Calculate size statistics
widths, heights = zip(*sizes)
min_width, max_width = min(widths), max(widths)
min_height, max_height = min(heights), max(heights)
avg_width = np.mean(widths)
avg_height = np.mean(heights)

# Print results
print("Количество изображений в каждом классе:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} изображений")
print(f"Минимальный размер: {min_width}x{min_height}")
print(f"Максимальный размер: {max_width}x{max_height}")
print(f"Средний размер: {avg_width:.2f}x{avg_height:.2f}")

# Visualization 1: Bar chart of class distribution
plt.figure(figsize=(12, 6))
bars = plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Классы', fontsize=12)
plt.ylabel('Количество изображений', fontsize=12)
plt.title('Распределение изображений по классам', fontsize=14)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'))
plt.close()

# Visualization 2: Histogram with KDE for size distribution
plt.figure(figsize=(12, 6))
sns.histplot(widths, bins=20, kde=True, color='blue', label='Ширина', stat='density')
sns.histplot(heights, bins=20, kde=True, color='orange', label='Высота', stat='density')
plt.xlabel('Размер (пиксели)', fontsize=12)
plt.ylabel('Плотность', fontsize=12)
plt.title('Распределение размеров изображений', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'size_distribution.png'))
plt.close()

# Visualization 3: Scatter plot of width vs height
df_sizes = pd.DataFrame({'Ширина': widths, 'Высота': heights})
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_sizes, x='Ширина', y='Высота', alpha=0.6)
plt.title('Соотношение ширины и высоты изображений', fontsize=14)
plt.xlabel('Ширина (пиксели)', fontsize=12)
plt.ylabel('Высота (пиксели)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'size_scatter.png'))
plt.close()

print(f"Графики сохранены в папке {RESULTS_DIR}")