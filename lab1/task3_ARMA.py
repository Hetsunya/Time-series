import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Генерация временного ряда
np.random.seed(42)
Y = np.cumsum(np.random.normal(0, 1, 100))

# Построение модели ARIMA
order = (2, 1, 1)  # Порядок авторегрессии (p), порядок интегрирования (d), порядок скользящего среднего (q)
model = sm.tsa.ARIMA(endog=Y, order=order)
results = model.fit()

# Прогнозирование
forecast = results.predict(start=len(Y), end=len(Y) + 10, dynamic=True, typ='levels')

# Вывод результатов
print(results.summary())

# Визуализация
plt.plot(Y, label='Оригинальный временной ряд')
plt.plot(np.arange(len(Y), len(Y) + 11), forecast, label='Прогноз')
plt.title('ARIMA модель и прогноз')
plt.xlabel('Временные шаги')
plt.ylabel('Значения')
plt.legend()
plt.show()
