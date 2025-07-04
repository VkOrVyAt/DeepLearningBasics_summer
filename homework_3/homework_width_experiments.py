import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import logging
from pathlib import Path

from utils.experiment_utils import fit
from utils.model_utils import MLP
from utils.visualization_utils import draw_curves, weight_hist
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создание директорий
Path('results/width_experiments').mkdir(parents=True, exist_ok=True)
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
    """Подсчет обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_width_experiments(epochs: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    width_configs = [
        [64, 32, 16],
        [256, 128, 64],
        [1024, 512, 256],
        [2048, 1024, 512]
    ]
    results = []

    # Эксперименты с фиксированной глубиной и разной шириной
    for hidden_sizes in width_configs:
        logger.info(f'Training model with hidden sizes {hidden_sizes}')
        model = MLP(hidden_sizes)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start = time.time()
        history = fit(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
        duration = time.time() - start

        params = count_parameters(model)
        results.append({
            'widths': hidden_sizes,
            'train_acc': history['train_acc'][-1],
            'test_acc': history['val_acc'][-1],
            'training_time': duration,
            'parameters': params
        })

        draw_curves(history, f'Width {hidden_sizes}', f'plots/width_{hidden_sizes}')
        weight_hist(model, f'Width {hidden_sizes} Weights', f'plots/width_{hidden_sizes}_weights')

    # Grid search оптимальной ширины
    layer1_sizes = [64, 256, 512]
    layer2_sizes = [32, 128, 256]
    layer3_sizes = [16, 64, 128]
    best = {'config': None, 'acc': 0.0}
    grid_results = []

    for l1, l2, l3 in itertools.product(layer1_sizes, layer2_sizes, layer3_sizes):
        hidden = [l1, l2, l3]
        logger.info(f'Grid search: {hidden}')
        model = MLP(hidden)
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # используем 5 эпох для grid search
        history_gs = fit(model, train_loader, test_loader, loss_fn, optimizer, 5, device)
        acc = history_gs['val_acc'][-1]
        grid_results.append({'widths': hidden, 'test_acc': acc})
        if acc > best['acc']:
            best = {'config': hidden, 'acc': acc}

    # Визуализация heatmap (фиксируем третий слой = 64)
    acc_matrix = []
    for l1 in layer1_sizes:
        row = []
        for l2 in layer2_sizes:
            entry = next((r['test_acc'] for r in grid_results if r['widths'] == [l1, l2, 64]), 0)
            row.append(entry)
        acc_matrix.append(row)

    plt.figure(figsize=(8, 6))
    sns.heatmap(acc_matrix, annot=True, xticklabels=layer2_sizes, yticklabels=layer1_sizes)
    plt.title('Grid Search Accuracy (3rd layer = 64)')
    plt.xlabel('Layer2 Size')
    plt.ylabel('Layer1 Size')
    plt.savefig('plots/grid_search_heatmap.png')
    plt.close()

    # Сохранение результатов
    with open('results/width_experiments/results.txt', 'w') as f:
        for r in results:
            f.write(f"Widths: {r['widths']}, Train Acc: {r['train_acc']:.4f}, Test Acc: {r['test_acc']:.4f}, "
                    f"Time: {r['training_time']:.2f}s, Params: {r['parameters']}\n")
        f.write(f"Best Grid Config: {best['config']}, Acc: {best['acc']:.4f}\n")


if __name__ == '__main__':
    run_width_experiments()
