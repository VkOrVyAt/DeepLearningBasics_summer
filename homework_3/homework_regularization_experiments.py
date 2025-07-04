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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создание директорий
Path('results/regularization_experiments').mkdir(parents=True, exist_ok=True)
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


def run_regularization_experiments(epochs: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = [
        {'name': 'No Reg', 'hidden_sizes': [256,128,64], 'use_dropout': False, 'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.0},
        {'name': 'Dropout 0.1', 'hidden_sizes': [256,128,64], 'use_dropout': True, 'dropout_rate': 0.1, 'use_batchnorm': False, 'weight_decay': 0.0},
        {'name': 'Dropout 0.3', 'hidden_sizes': [256,128,64], 'use_dropout': True, 'dropout_rate': 0.3, 'use_batchnorm': False, 'weight_decay': 0.0},
        {'name': 'Dropout 0.5', 'hidden_sizes': [256,128,64], 'use_dropout': True, 'dropout_rate': 0.5, 'use_batchnorm': False, 'weight_decay': 0.0},
        {'name': 'BatchNorm',  'hidden_sizes': [256,128,64], 'use_dropout': False, 'dropout_rate': 0.0, 'use_batchnorm': True,  'weight_decay': 0.0},
        {'name': 'Dropout+BN', 'hidden_sizes': [256,128,64], 'use_dropout': True, 'dropout_rate': 0.3, 'use_batchnorm': True,  'weight_decay': 0.0},
        {'name': 'L2 0.01',   'hidden_sizes': [256,128,64], 'use_dropout': False, 'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.01}
    ]
    results = []

    # Сравнение техник регуляризации
    for cfg in configs:
        logger.info(f"Training {cfg['name']}")
        model = MLP(hidden_sizes=cfg['hidden_sizes'], use_dropout=cfg['use_dropout'], dropout_rate=cfg['dropout_rate'], use_bn=cfg['use_batchnorm'])
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=cfg['weight_decay'])

        start = time.time()
        history = fit(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
        duration = time.time() - start

        train_acc = history['train_acc'][-1]
        test_acc = history['val_acc'][-1]
        params = count_parameters(model)
        results.append({'name': cfg['name'], 'train_acc': train_acc, 'test_acc': test_acc, 'time': duration, 'params': params})

        draw_curves(history, f"{cfg['name']}", f"plots/reg_{cfg['name'].replace(' ','_')}")
        weight_hist(model, f"{cfg['name']} Weights", f"plots/reg_{cfg['name'].replace(' ','_')}_weights")

    # Адаптивная регуляризация: dropout изменяется по слоям
    adaptive_cfg = {'name': 'Adaptive Dropout', 'hidden_sizes': [256,128,64], 'use_dropout': True, 'dropout_rate': 0.5, 'use_batchnorm': False, 'weight_decay': 0.0}
    logger.info(f"Training {adaptive_cfg['name']}")
    model = MLP(hidden_sizes=adaptive_cfg['hidden_sizes'], use_dropout=True, dropout_rate=adaptive_cfg['dropout_rate'], use_bn=False)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history_ad = fit(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
    draw_curves(history_ad, adaptive_cfg['name'], f"plots/{adaptive_cfg['name'].replace(' ','_')}")

    # Сохранение результатов
    with open('results/regularization_experiments/results.txt', 'w') as f:
        for r in results:
            f.write(f"{r['name']}: Train Acc={r['train_acc']:.4f}, Test Acc={r['test_acc']:.4f}, Time={r['time']:.2f}s, Params={r['params']}\n")
        f.write(f"Adaptive Dropout: Test Acc={history_ad['val_acc'][-1]:.4f}\n")


if __name__ == '__main__':
    run_regularization_experiments()
