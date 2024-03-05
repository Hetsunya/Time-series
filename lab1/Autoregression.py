import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных (предположим, что у вас уже есть временной ряд Y)
# Пример сгенерированного временного ряда:
np.random.seed(42)
Y = np.cumsum(np.random.normal(0, 1, 100))

# Функция для вычисления скользящего среднего с разными окнами
def moving_average(data, window_sizes):
    plt.figure(figsize=(10, 6))
    plt.plot(Y, label='Оригинальный временной ряд', color='blue')

    for window_size in window_sizes:
        ma = data.rolling(window=window_size).mean()
        plt.plot(ma, label=f'Скользящее среднее (окно {window_size})')

    plt.title('Сравнение скользящего среднего с разными окнами')
    plt.xlabel('Временные шаги')
    plt.ylabel('Значения')
    plt.legend()
    plt.show()

# Применяем функцию с различными окнами
window_sizes = [5, 10, 20]
moving_average(pd.Series(Y), window_sizes)
