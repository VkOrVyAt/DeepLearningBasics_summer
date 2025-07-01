import torch
import numpy as np

# 2.1 soft calcultation gradient
# Создайте тензоры x, y, z с requires_grad=True
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
# Найдите градиенты по всем переменным
# Проверьте результат аналитически

# Создаем тензоры с requires_grad=True для отслеживания градиентов
# requires_grad=True позволяет PyTorch отслеживать операции с этими тензорами, для последующего вычисления градиентов
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

# Определяем функцию
def f(x, y, z):
    return x**2 + y**2 + z**2 + 2*x*y*z

# Вычисляем значение функции
result = f(x, y, z)
# Вычисляем градиенты
result.backward()

# Получаем градиенты
grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

print("\n2.1 Function value:", result.item())
print("Gradient with respect to x:", grad_x.item())
print("Gradient with respect to y:", grad_y.item())
print("Gradient with respect to z:", grad_z.item())
# Проверяем аналитически
print("\n2.1 Analytical gradients:")
print("d(f)/dx = 2*x + 2*y*z =", 2*x.item() + 2*y.item()*z.item())
print("d(f)/dy = 2*y + 2*x*z =", 2*y.item() + 2*x.item()*z.item())
print("d(f)/dz = 2*z + 2*x*y =", 2*z.item() + 2*x.item()*y.item())

# 2.2 gradient descent
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b

# Определяем функцию потерь MSE
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# Примерные данные
x_data = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# Параметры модели с requires_grad=True
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Линейная функция
y_pred = w * x_data + b

# Вычисляем значение функции потерь
loss = mse_loss(y_pred, y_true)

# Вычисляем градиенты
loss.backward()

# Получаем градиенты
grad_w = w.grad
grad_b = b.grad

print("\n2.2 Loss value:", loss.item())
print("Gradient with respect to w:", grad_w.item())
print("Gradient with respect to b:", grad_b.item())

# Аналитическая проверка
n = len(x_data)
errors = y_pred - y_true
grad_w_analytical = (2 / n) * torch.sum(errors * x_data)
grad_b_analytical = (2 / n) * torch.sum(errors)

print("\n2.2 Analytical gradients:")
print("d(MSE)/dw = (2/n) * Σ[(y_pred - y_true) * x] =", grad_w_analytical.item())
print("d(MSE)/db = (2/n) * Σ(y_pred - y_true) =", grad_b_analytical.item())

# 2.3
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad

# Определяем функцию f(x) = sin(x^2 + 1)
f = lambda x: torch.sin(x**2 + 1)

# Создаем тензор x с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)

# Вычисляем значение функции
result = f(x)

# Вычисляем градиент с помощью torch.autograd.grad
# torch.autograd.grad возвращает кортеж, поэтому берем первый элемент
grad_x = torch.autograd.grad(result, x)[0]

print("\n2.3 Значение функции:", result.item())
print("Градиент по x (PyTorch):", grad_x.item())

# Аналитический градиент: df/dx = 2x * cos(x^2 + 1)
analytical_grad = 2 * x * torch.cos(x**2 + 1)
print("Аналитический градиент:", analytical_grad.item())

# Проверяем, что градиенты совпадают (разница должна быть близка к 0)
difference = (grad_x - analytical_grad).item()
print("Разница между градиентами:", difference)