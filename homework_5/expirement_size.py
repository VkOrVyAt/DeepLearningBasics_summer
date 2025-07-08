import os
import time
import psutil
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import gc
from torchvision import transforms
from datasets import CustomImageDataset
from PIL import Image

# Пути к данным и результатам
DATA_DIR = 'D:/ML/homework_5/data/train'  # Папка с тренировочными данными
RESULTS_DIR = 'D:/ML/homework_5/result/task5'  # Папка для результатов
os.makedirs(RESULTS_DIR, exist_ok=True)  # Создаем папку, если не существует

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None)  # Датасет без трансформаций

# Выбор 100 случайных изображений
random.seed(42)  # Фиксируем seed для воспроизводимости
indices = random.sample(range(len(dataset)), 100)  # Случайные индексы
selected_dataset = torch.utils.data.Subset(dataset, indices)  # Подмножество датасета

# Конфигурация аугментаций
aug_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Изменение яркости/контраста
    transforms.RandomRotation(degrees=15)  # Случайный поворот
])

# Функция для измерения времени и памяти
def measure_performance(size, num_runs=5):
    """Измеряет время и память для обработки изображений заданного размера."""
    times = []
    mem_usages = []
    
    for _ in range(num_runs):  # Многократный запуск для точности
        gc.collect()  # Очищаем память перед замером
        process = psutil.Process(os.getpid())  # Текущий процесс
        mem_before = process.memory_info().rss / 1024 ** 2  # Память до (МБ)
        start_time = time.time()  # Начало замера времени
        
        # Обработка изображений
        transform_resize = transforms.Compose([
            transforms.Resize(size),  # Изменение размера
            aug_pipeline  # Применение аугментаций
        ])
        for img, _ in selected_dataset:
            img = transform_resize(img)  # Применяем ресайз и аугментации
        
        end_time = time.time()  # Конец замера времени
        mem_after = process.memory_info().rss / 1024 ** 2  # Память после (МБ)
        
        times.append(end_time - start_time)  # Время обработки
        mem_usages.append(max(0, mem_after - mem_before))  # Использование памяти
    
    return np.mean(times), np.mean(mem_usages)  # Средние значения

# Эксперимент с разными размерами
sizes = [(64, 64), (128, 128), (224, 224), (512, 512), (1024, 1024)]  # Размеры изображений
times = []
memories = []

for size in sizes:
    print(f"Обработка размера {size[0]}x{size[1]}...")
    load_time, mem_usage = measure_performance(size)  # Замеряем производительность
    times.append(load_time)
    memories.append(mem_usage)
    print(f"Время: {load_time:.2f} сек, Память: {mem_usage:.2f} МБ")

# Визуализация результатов
plt.style.use('ggplot')  # Стиль для красивых графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# График времени
ax1.plot([s[0] for s in sizes], times, marker='o', color='#1E90FF', linewidth=2)
for i, (size, t) in enumerate(zip(sizes, times)):
    ax1.annotate(f'{t:.2f} с', (size[0], t), textcoords="offset points", xytext=(0,10), ha='center')
ax1.set_xscale('log')  # Логарифмическая шкала для размеров
ax1.set_xlabel('Размер изображения (пиксели)')
ax1.set_ylabel('Время обработки (сек)')
ax1.set_title('Зависимость времени от размера')
ax1.grid(True)

# График памяти
ax2.plot([s[0] for s in sizes], memories, marker='o', color='#FF4500', linewidth=2)
for i, (size, m) in enumerate(zip(sizes, memories)):
    ax2.annotate(f'{m:.2f} МБ', (size[0], m), textcoords="offset points", xytext=(0,10), ha='center')
ax2.set_xscale('log')  # Логарифмическая шкала для размеров
ax2.set_xlabel('Размер изображения (пиксели)')
ax2.set_ylabel('Использование памяти (МБ)')
ax2.set_title('Зависимость памяти от размера')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'task5_performance.png'), dpi=150)
plt.close()

print(f"Графики сохранены в {RESULTS_DIR}")