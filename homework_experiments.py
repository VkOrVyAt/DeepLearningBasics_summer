import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from homework_model_modification import LogisticRegression, train_model, compute_metrics
from homework_datasets import CustomCSVDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Часть 3.1: Исследование гиперпараметров
def hyperparameter_tuning():
    """Проведение экспериментов с гиперпараметрами и визуализация результатов."""
    # Параметры для экспериментов
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']
    
    # Загрузка датасета Titanic
    dataset = CustomCSVDataset(
        csv_path='data/Titanic-Dataset.csv',
        target_col='Survived',
        task='classification',
        cat_cols=['Sex', 'Embarked', 'Pclass'],
        bin_cols=[],
        drop_cols=['Name', 'Ticket', 'Cabin', 'PassengerId']
    )
    
    results = []
    
    # Перебор всех комбинаций гиперпараметров
    for lr in learning_rates:
        for bs in batch_sizes:
            for opt_name in optimizers:
                logger.info(f"Эксперимент: lr={lr}, batch_size={bs}, optimizer={opt_name}")
                
                # Создание DataLoader с текущим размером батча
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
                
                # Инициализация модели
                model = LogisticRegression(in_features=dataset.features.shape[1], num_classes=2)
                criterion = nn.CrossEntropyLoss()
                
                # Выбор оптимизатора
                if opt_name == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                elif opt_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                else:  # RMSprop
                    optimizer = optim.RMSprop(model.parameters(), lr=lr)
                
                # Обучение модели
                train_model(model, dataloader, criterion, optimizer, epochs=50, patience=5)
                
                # Оценка модели
                model.eval()
                with torch.no_grad():
                    y_pred = model(dataset.features)
                    y_prob = y_pred.numpy()
                    y_pred = torch.argmax(y_pred, dim=1).numpy()
                    y_true = dataset.targets.numpy()
                
                precision, recall, f1, roc_auc = compute_metrics(y_true, y_pred, y_prob, num_classes=2)
                
                # Сохранение результатов
                results.append({
                    'learning_rate': lr,
                    'batch_size': bs,
                    'optimizer': opt_name,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                })
    
    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/hyperparameter_results.csv', index=False)
    logger.info("Результаты сохранены в data/hyperparameter_results.csv")
    
    # Визуализация: график F1-score в зависимости от learning rate для каждого оптимизатора
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='learning_rate', y='f1', hue='optimizer', style='batch_size', markers=True)
    plt.title('F1-score в зависимости от Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('F1-score')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/hyperparameter_f1.png')
    plt.close()
    logger.info("График сохранен в plots/hyperparameter_f1.png")

# Часть 3.2: Feature Engineering
def feature_engineering():
    """Создание новых признаков и сравнение с базовой моделью."""
    # Загрузка исходного датасета Titanic
    df = pd.read_csv('data/Titanic-Dataset.csv')
    
    # Создание новых признаков
    df['Age_squared'] = df['Age'] ** 2  # Полиномиальный признак
    df['Fare_times_Age'] = df['Fare'] * df['Age']  # Взаимодействие между признаками
    df['Pclass_mean_fare'] = df.groupby('Pclass')['Fare'].transform('mean')  # Статистический признак
    
    # Сохранение улучшенного датасета
    enhanced_csv_path = 'data/titanic_enhanced.csv'
    df.to_csv(enhanced_csv_path, index=False)
    logger.info(f"Улучшенный датасет сохранен в {enhanced_csv_path}")
    
    # Базовый датасет
    base_dataset = CustomCSVDataset(
        csv_path='data/Titanic-Dataset.csv',
        target_col='Survived',
        task='classification',
        cat_cols=['Sex', 'Embarked', 'Pclass'],
        bin_cols=[],
        drop_cols=['Name', 'Ticket', 'Cabin', 'PassengerId']
    )
    base_dataloader = DataLoader(base_dataset, batch_size=32, shuffle=True)
    
    # Улучшенный датасет
    enhanced_dataset = CustomCSVDataset(
        csv_path=enhanced_csv_path,
        target_col='Survived',
        task='classification',
        cat_cols=['Sex', 'Embarked', 'Pclass'],
        bin_cols=[],
        drop_cols=['Name', 'Ticket', 'Cabin', 'PassengerId']
    )
    enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=32, shuffle=True)
    
    # Обучение базовой модели
    base_model = LogisticRegression(in_features=base_dataset.features.shape[1], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.01)
    train_model(base_model, base_dataloader, criterion, optimizer, epochs=50, patience=5)
    
    base_model.eval()
    with torch.no_grad():
        y_pred_base = base_model(base_dataset.features)
        y_prob_base = y_pred_base.numpy()
        y_pred_base = torch.argmax(y_pred_base, dim=1).numpy()
        y_true_base = base_dataset.targets.numpy()
    base_metrics = compute_metrics(y_true_base, y_pred_base, y_prob_base, num_classes=2)
    
    # Обучение модели на улучшенном датасете
    enhanced_model = LogisticRegression(in_features=enhanced_dataset.features.shape[1], num_classes=2)
    optimizer = optim.Adam(enhanced_model.parameters(), lr=0.01)
    train_model(enhanced_model, enhanced_dataloader, criterion, optimizer, epochs=50, patience=5)
    
    enhanced_model.eval()
    with torch.no_grad():
        y_pred_enhanced = enhanced_model(enhanced_dataset.features)
        y_prob_enhanced = y_pred_enhanced.numpy()
        y_pred_enhanced = torch.argmax(y_pred_enhanced, dim=1).numpy()
        y_true_enhanced = enhanced_dataset.targets.numpy()
    enhanced_metrics = compute_metrics(y_true_enhanced, y_pred_enhanced, y_prob_enhanced, num_classes=2)
    
    # Сравнение метрик
    comparison = pd.DataFrame({
        'Model': ['Base', 'Enhanced'],
        'F1-score': [base_metrics[2], enhanced_metrics[2]],
        'ROC-AUC': [base_metrics[3], enhanced_metrics[3]]
    })
    logger.info(f"Сравнение:\n{comparison}")
    
    # Визуализация сравнения
    plt.figure(figsize=(8, 5))
    comparison.set_index('Model')[['F1-score', 'ROC-AUC']].plot(kind='bar')
    plt.title('Сравнение базовой и улучшенной моделей')
    plt.ylabel('Значение метрики')
    plt.savefig('plots/feature_engineering_comparison.png')
    plt.close()
    logger.info("Сравнительный график сохранен в plots/feature_engineering_comparison.png")

if __name__ == '__main__':
    hyperparameter_tuning()
    feature_engineering()