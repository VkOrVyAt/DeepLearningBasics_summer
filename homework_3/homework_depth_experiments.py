import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import logging
import os
from pathlib import Path

from utils.experiment_utils import fit, validate
from utils.model_utils import MLP_1
from utils.visualization_utils import draw_curves, weight_hist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создание директорий для результатов и графиков
Path('results/depth_experiments').mkdir(parents=True, exist_ok=True)
Path('plots').mkdir(exist_ok=True)

# Подготовка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


def count_parameters(model: nn.Module) -> int:
    """Подсчет числа обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_depth_experiments(epochs: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depths = [1, 2, 3, 5, 7]
    results = []

    # Эксперименты без регуляризации
    for num_layers in depths:
        logger.info(f'Training model with {num_layers} layers')
        model = MLP_1(num_layers=num_layers)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_time = time.time()
        history = fit(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
        training_time = time.time() - start_time

        params = count_parameters(model)
        results.append({
            'depth': num_layers,
            'train_acc': history['train_acc'][-1],
            'test_acc': history['val_acc'][-1],
            'training_time': training_time,
            'parameters': params
        })

        draw_curves(history, f'MLP {num_layers} Layers', f'plots/depth_{num_layers}')

    # Эксперименты с Dropout и BatchNorm
    for num_layers in depths:
        logger.info(f'Training model with {num_layers} layers + Dropout & BatchNorm')
        model = MLP_1(num_layers=num_layers, use_dropout=True, dropout_rate=0.3, use_batchnorm=True)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_time = time.time()
        history = fit(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
        training_time = time.time() - start_time

        params = count_parameters(model)
        results.append({
            'depth': num_layers,
            'train_acc': history['train_acc'][-1],
            'test_acc': history['val_acc'][-1],
            'training_time': training_time,
            'parameters': params,
            'dropout_batchnorm': True
        })

        draw_curves(history, f'MLP {num_layers} Layers with Dropout+BN', f'plots/depth_{num_layers}_dropout_bn')
        # Распределение весов
        weight_hist(model, f'Depth {num_layers} Weight Distribution', f'plots/depth_{num_layers}_weights')

    # Сохранение результатов в файл
    result_path = Path('results/depth_experiments/results.txt')
    with result_path.open('w') as f:
        for r in results:
            parts = [f"Depth: {r['depth']}",
                     f"Train Acc: {r['train_acc']:.4f}",
                     f"Test Acc: {r['test_acc']:.4f}",
                     f"Time: {r['training_time']:.2f}s",
                     f"Params: {r['parameters']}"]
            if r.get('dropout_batchnorm'):
                parts.append('Dropout+BN')
            f.write(', '.join(parts) + '\n')


if __name__ == '__main__':
    run_depth_experiments()




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import time
# import logging
# import os
# from pathlib import Path

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Создание директорий для результатов и графиков
# Path('results/depth_experiments').mkdir(parents=True, exist_ok=True)
# Path('plots').mkdir(exist_ok=True)

# # Подготовка данных MNIST
# # transforms.ToTensor() преобразует изображения в тензоры, Normalize нормализует значения пикселей
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# class MLP(nn.Module):
#     # Класс для создания многослойного перцептрона с заданным числом слоев
#     def __init__(self, num_layers, hidden_size=128, use_dropout=False, dropout_rate=0.3, use_batchnorm=False):
#         super(MLP, self).__init__()
#         layers = []
#         input_size = 784  # Размер входа для MNIST (28x28)
#         output_size = 10  # Количество классов
#         # Для одного слоя создается только линейный классификатор
#         if num_layers == 1:
#             layers.append(nn.Linear(input_size, output_size))
#         else:
#             # Первый слой: вход -> скрытый слой
#             layers.append(nn.Linear(input_size, hidden_size))
#             if use_batchnorm:
#                 layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm нормализует активации
#             layers.append(nn.ReLU())  # ReLU добавляет нелинейность
#             if use_dropout:
#                 layers.append(nn.Dropout(dropout_rate))  # Dropout предотвращает переобучение
#             # Промежуточные скрытые слои
#             for _ in range(num_layers - 2):
#                 layers.append(nn.Linear(hidden_size, hidden_size))
#                 if use_batchnorm:
#                     layers.append(nn.BatchNorm1d(hidden_size))
#                 layers.append(nn.ReLU())
#                 if use_dropout:
#                     layers.append(nn.Dropout(dropout_rate))
#             # Последний слой: скрытый -> выход
#             layers.append(nn.Linear(hidden_size, output_size))
#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         # Преобразование входного изображения в одномерный вектор
#         x = x.view(-1, 784)
#         return self.model(x)

# def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
#     # Функция для обучения модели
#     model = model.to(device)
#     criterion = nn.NLLLoss()  # Функция потерь для классификации
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор Adam
#     train_losses, test_losses, train_accs, test_accs = [], [], [], []
    
#     for epoch in range(epochs):
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         start_time = time.time()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()  # Обнуление градиентов
#             outputs = model(images)
#             outputs = torch.log_softmax(outputs, dim=1)  # Применение log_softmax
#             loss = criterion(outputs, labels)
#             loss.backward()  # Обратное распространение
#             optimizer.step()  # Обновление весов
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         train_loss = running_loss / len(train_loader)
#         train_acc = correct / total
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
        
#         # Оценка на тестовом наборе
#         model.eval()
#         test_loss, correct, total = 0.0, 0, 0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 outputs = torch.log_softmax(outputs, dim=1)
#                 test_loss += criterion(outputs, labels).item()
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         test_loss = test_loss / len(test_loader)
#         test_acc = correct / total
#         test_losses.append(test_loss)
#         test_accs.append(test_acc)
        
#         logger.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
#     training_time = time.time() - start_time
#     return train_losses, test_losses, train_accs, test_accs, training_time

# def plot_learning_curves(train_losses, test_losses, train_accs, test_accs, title, filename):
#     # Функция для построения кривых обучения
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(test_losses, label='Test Loss')
#     plt.title(f'{title} - Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(train_accs, label='Train Accuracy')
#     plt.plot(test_accs, label='Test Accuracy')
#     plt.title(f'{title} - Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'plots/{filename}.png')
#     plt.close()

# def count_parameters(model):
#     # Подсчет количества параметров модели
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def run_depth_experiments():
#     # Основная функция для экспериментов с глубиной
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     depths = [1, 2, 3, 5, 7]
#     results = []
    
#     # Эксперименты без регуляризации
#     for num_layers in depths:
#         logger.info(f'Training model with {num_layers} layers')
#         model = MLP(num_layers=num_layers)
#         train_losses, test_losses, train_accs, test_accs, training_time = train_model(model, train_loader, test_loader, device=device)
#         params = count_parameters(model)
#         results.append({
#             'depth': num_layers,
#             'train_acc': train_accs[-1],
#             'test_acc': test_accs[-1],
#             'training_time': training_time,
#             'parameters': params
#         })
#         plot_learning_curves(train_losses, test_losses, train_accs, test_accs, f'MLP {num_layers} Layers', f'depth_{num_layers}')
    
#     # Эксперименты с Dropout и BatchNorm
#     for num_layers in depths:
#         logger.info(f'Training model with {num_layers} layers, Dropout and BatchNorm')
#         model = MLP(num_layers=num_layers, use_dropout=True, dropout_rate=0.3, use_batchnorm=True)
#         train_losses, test_losses, train_accs, test_accs, training_time = train_model(model, train_loader, test_loader, device=device)
#         params = count_parameters(model)
#         results.append({
#             'depth': num_layers,
#             'train_acc': train_accs[-1],
#             'test_acc': test_accs[-1],
#             'training_time': training_time,
#             'parameters': params,
#             'dropout_batchnorm': True
#         })
#         plot_learning_curves(train_losses, test_losses, train_accs, test_accs, f'MLP {num_layers} Layers with Dropout and BatchNorm', f'depth_{num_layers}_dropout_bn')
    
#     # Сохранение результатов
#     with open('results/depth_experiments/results.txt', 'w') as f:
#         for result in results:
#             f.write(f"Depth: {result['depth']}, Train Acc: {result['train_acc']:.4f}, Test Acc: {result['test_acc']:.4f}, Time: {result['training_time']:.2f}s, Params: {result['parameters']}\n")

# if __name__ == '__main__':
#     run_depth_experiments()