import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_curves(history, title: str, out_path: str, extension: str = 'png'):
    """
    Построение и сохранение кривых обучения.

    history: dict с ключами 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    title: заголовок графиков
    out_path: путь без расширения
    extension: формат изображения (по умолчанию 'png')
    """
    # Добавляем расширение
    save_file = f"{out_path}.{extension}" if not out_path.endswith(f".{extension}") else out_path

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"{title} - Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f"{title} - Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()


def weight_hist(model, title: str, out_path: str, extension: str = 'png'):
    """
    Построение и сохранение гистограммы распределения весов модели.

    model: nn.Module
    title: заголовок графика
    out_path: путь без расширения
    extension: формат изображения (по умолчанию 'png')
    """
    save_file = f"{out_path}.{extension}" if not out_path.endswith(f".{extension}") else out_path

    # Собираем все веса
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.data.cpu().numpy().flatten())
    if all_weights:
        weights = np.concatenate(all_weights)
    else:
        weights = np.array([])

    plt.figure(figsize=(8, 6))
    sns.histplot(weights, bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()
