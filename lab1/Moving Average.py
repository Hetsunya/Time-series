import numpy as np
import matplotlib.pyplot as plt

# Создаем случайную временную серию
np.random.seed(42)
time_series = np.random.rand(50) * 10  # случайные значения от 0 до 10

# Функция для вычисления скользящего среднего
def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Задаем размер окна для скользящего среднего
window_size = 3

# Вычисляем скользящее среднее
ma_values = moving_average(time_series, window_size)

# Визуализируем результаты
plt.plot(time_series, label='Исходная временная серия')
plt.plot(np.arange(window_size - 1, len(time_series)), ma_values, label=f'Скользящее среднее ({window_size}-период)')
plt.legend()
plt.title('Пример скользящего среднего')
plt.xlabel('Временные точки')
plt.ylabel('Значения')
plt.show()
