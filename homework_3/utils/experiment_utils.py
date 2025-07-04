import torch
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger("trainer")


def fit(model, train_loader: DataLoader, val_loader: DataLoader, loss_fn, optimizer, num_epochs: int, device):
    """Обучает модель и возвращает историю метрик."""
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        model.to(device).train()
        epoch_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x.view(batch_x.size(0), -1))
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            _, preds_labels = preds.max(dim=1)
            correct += (preds_labels == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total
        val_loss, val_acc = validate(model, val_loader, device, loss_fn)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        logger.info(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}")

    return history


def validate(model, data_loader: DataLoader, device, loss_fn=None):
    """Оценивает модель на валидационном наборе."""
    model.to(device).eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x.view(batch_x.size(0), -1))
            if loss_fn:
                total_loss += loss_fn(outputs, batch_y).item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total if loss_fn else 0.0
    accuracy = correct / total
    return avg_loss, accuracy