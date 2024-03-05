import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Параметры модели
c = 2
phi = np.array([0.5, -0.2])  # Пример коэффициентов авторегрессии
theta = np.array([0.3, -0.1])  # Пример коэффициентов скользящего среднего
sigma = 0.1  # Стандартное отклонение ошибки

# Генерация временного ряда
n_samples = 100
time_series = np.zeros(n_samples)

for t in range(2, n_samples):
    autoregressive = np.dot(phi, time_series[t-2:t][::-1])
    moving_average = np.dot(theta, np.random.normal(0, sigma, len(theta)))
    time_series[t] = c + autoregressive + moving_average

# Построение графика
plt.figure(figsize=(15, 5))
plt.plot(time_series, label='Generated Time Series')
plt.title('Generated Time Series using ARMA(p, q) Equation')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()
