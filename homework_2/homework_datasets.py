# 2.1 Кастомный Dataset класс
# # Создайте кастомный класс датасета для работы с CSV файлами:
# # - Загрузка данных из файла
# # - Предобработка (нормализация, кодирование категорий)
# # - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)

# 2.2 Эксперименты с различными датасетами
# # Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

import os
import logging
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

from homework_2.homework_model_modification import LinearRegression, LogisticRegression, train_model, compute_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CustomCSVDataset(Dataset):
    def __init__(self, csv_path, target_col, task='classification', cat_cols=None, bin_cols=None, drop_cols=None):
        """
        Кастомный класс датасета для работы с CSV-файлами.

        Аргументы:
            csv_path (str): Путь к CSV-файлу.
            target_col (str): Название целевого столбца.
            task (str): Тип задачи ('classification' или 'regression').
            cat_cols (list): Список категориальных столбцов.
            bin_cols (list): Список бинарных столбцов.
            drop_cols (list): Список столбцов для удаления.
        """
        self.df = pd.read_csv(csv_path)
        self.target_col = target_col
        self.task = task
        self.cat_cols = cat_cols or []
        self.bin_cols = bin_cols or []
        self.drop_cols = drop_cols or []

        # Предобработка данных
        self._preprocess_data()

    def _preprocess_data(self):
        """Выполняет предобработку данных."""
        # Удаление ненужных столбцов
        if self.drop_cols:
            self.df = self.df.drop(columns=self.drop_cols)

        # Обработка пропущенных значений
        # Категориальные столбцы: заполняем модой (самым частым значением)
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Бинарные столбцы: заполняем модой
        for col in self.bin_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Числовые столбцы: заполняем медианой
        num_cols = [col for col in self.df.columns 
                    if col not in self.cat_cols + self.bin_cols + [self.target_col]]
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Кодирование категориальных столбцов
        self.encoders = {}
        for col in self.cat_cols:
            self.encoders[col] = LabelEncoder()
            self.df[col] = self.encoders[col].fit_transform(self.df[col])

        # Обработка бинарных столбцов (преобразование в 0 и 1)
        for col in self.bin_cols:
            self.df[col] = self.df[col].apply(lambda x: 1 if x > 0 else 0)

        # Нормализация числовых столбцов
        self.scaler = StandardScaler()
        if num_cols:
            self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])

        # Разделение на признаки и целевую переменную
        self.features = self.df.drop(columns=[self.target_col]).values
        self.targets = self.df[self.target_col].values

        # Преобразование в тензоры PyTorch
        self.features = torch.tensor(self.features, dtype=torch.float32)
        if self.task == 'classification':
            self.targets = torch.tensor(self.targets, dtype=torch.long)
        else:
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

# Эксперимент с регрессией (Boston Housing)
def experiment_regression():
    """Эксперимент с регрессией на датасете Boston Housing."""
    dataset = CustomCSVDataset(
        csv_path='data/HousingData.csv',
        target_col='MEDV',
        task='regression',
        cat_cols=[],
        bin_cols=['CHAS'],  # Бинарный столбец
        drop_cols=[]
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LinearRegression(in_features=dataset.features.shape[1], l1_lambda=0.01, l2_lambda=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Обучение модели
    train_model(model, dataloader, criterion, optimizer, epochs=100, patience=10)
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/linreg_boston.pth')
    logger.info("Модель линейной регрессии сохранена в models/linreg_boston.pth")

# Эксперимент с классификацией (Titanic)
def experiment_classification():
    """Эксперимент с бинарной классификацией на датасете Titanic."""
    dataset = CustomCSVDataset(
        csv_path='data/Titanic-Dataset.csv',
        target_col='Survived',
        task='classification',
        cat_cols=['Sex', 'Embarked', 'Pclass'],
        bin_cols=[],
        drop_cols=['Name', 'Ticket', 'Cabin', 'PassengerId']  # Удаляем ненужные столбцы
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LogisticRegression(in_features=dataset.features.shape[1], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Обучение модели
    train_model(model, dataloader, criterion, optimizer, epochs=100, patience=10)
    
    # Оценка модели
    model.eval()
    with torch.no_grad():
        y_pred = model(dataset.features)
        y_prob = y_pred.numpy()
        y_pred = torch.argmax(y_pred, dim=1).numpy()
        y_true = dataset.targets.numpy()
    
    # Вычисление метрик и построение матрицы ошибок
    compute_metrics(y_true, y_pred, y_prob, num_classes=2)
    os.makedirs('plots', exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, num_classes=2, filename='plots/confusion_matrix_titanic.png')
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/logreg_titanic.pth')
    logger.info("Модель логистической регрессии сохранена в models/logreg_titanic.pth")

if __name__ == '__main__':
    # Запуск экспериментов
    experiment_regression()
    experiment_classification()