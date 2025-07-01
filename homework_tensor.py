import torch
import numpy as np

# 1.1 Create tensor
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
# - Тензор размером 2x3x4, заполненный нулями
# - Тензор размером 5x5, заполненный единицами
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)

tensor_3x4 = torch.rand(3, 4)
tensor_2x3x4 = torch.zeros(2, 3, 4)
tensor_5x5 = torch.ones(5, 5)
tensor_4x4 = torch.arange(16).reshape(4, 4)

print("1.1 Tensors created:")
print("Tensor 3x4:\n", tensor_3x4)
print("Tensor 2x3x4:\n", tensor_2x3x4)
print("Tensor 5x5:\n", tensor_5x5)
print("Tensor 4x4:\n", tensor_4x4)

# 1.2 Tensor operations
# Дано: тензор A размером 3x4 и тензор B размером 4x3
# Выполните:
# - Транспонирование тензора A
# - Матричное умножение A и B
# - Поэлементное умножение A и транспонированного B
# - Вычислите сумму всех элементов тензора A

tensor_A = torch.rand(3, 4)
tensor_B = torch.rand(4, 3)

tensor_A_transposed = tensor_A.T
tensor_product = torch.matmul(tensor_A, tensor_B) # or torch.mm(tensor_A, tensor_B) or tensor_A @ tensor_B or tensor_A.mm(tensor_B)
tensor_mult_elementwise = tensor_A * tensor_B.T
tensor_A_sum = tensor_A.sum()

print("\n1.2 Tensor A:\n", tensor_A)
print("\n1.2 Tensor A Transposed:\n", tensor_A_transposed)
print("\n1.2 Tensor B:\n", tensor_B)
print("\n1.2 Matrix multiplication of A and B:\n", tensor_product)
print("\n1.2 Element-wise multiplication of A and Transposed B:\n", tensor_mult_elementwise)
print("\n1.2 Sum of all elements in Tensor A:", tensor_A_sum)

# 1.3 Tensor indexing and slicing
# Создайте тензор размером 5x5x5
# Извлеките:
# - Первую строку
# - Последний столбец
# - Подматрицу размером 2x2 из центра тензора
# - Все элементы с четными индексами

tensor_5x5x5 = torch.rand(5, 5, 5)
print("\n1.3 Tensor 5x5x5:\n", tensor_5x5x5)

first_row = tensor_5x5x5[0, :, :]
last_column = tensor_5x5x5[:, -1, :]
center_submatrix = tensor_5x5x5[1:3, 1:3, :]
even_indexed_elements = tensor_5x5x5[::2, ::2, ::2]

print("\n1.3 First row:\n", first_row)
print("\n1.3 Last column:\n", last_column)
print("\n1.3 Center submatrix (2x2):\n", center_submatrix)
print("\n1.3 Even indexed elements:\n", even_indexed_elements)

# 1.4 Tensor reshaping and concatenation
# Создайте тензор размером 24 элемента
# Преобразуйте его в формы:
# - 2x12
# - 3x8
# - 4x6
# - 2x3x4
# - 2x2x2x3

tensor_24 = torch.arange(24)
reshaped_2x12 = tensor_24.reshape(2, 12)
reshaped_3x8 = tensor_24.reshape(3, 8)
reshaped_4x6 = tensor_24.reshape(4, 6)
reshaped_2x3x4 = tensor_24.reshape(2, 3, 4)
reshaped_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
print("\n1.4 Tensor 24 elements:\n", tensor_24)
print("\n1.4 Reshaped to 2x12:\n", reshaped_2x12)
print("\n1.4 Reshaped to 3x8:\n", reshaped_3x8)
print("\n1.4 Reshaped to 4x6:\n", reshaped_4x6)
print("\n1.4 Reshaped to 2x3x4:\n", reshaped_2x3x4)
print("\n1.4 Reshaped to 2x2x2x3:\n", reshaped_2x2x2x3)

