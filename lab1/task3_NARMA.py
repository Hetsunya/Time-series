import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Генерация временного ряда
np.random.seed(42)
Y = np.cumsum(np.random.normal(0, 1, 100))

# Построение параметрической NARMA
degree = 2
X = np.arange(1, len(Y) + 1).reshape(-1, 1)
X_poly = PolynomialFeatures(degree).fit_transform(X)
model = make_pipeline(LinearRegression())
model.fit(X_poly, Y)

# Прогнозирование
forecast = model.predict(PolynomialFeatures(degree).fit_transform(np.arange(len(Y), len(Y) + 11).reshape(-1, 1)))

# Визуализация
plt.plot(Y, label='Оригинальный временной ряд')
plt.plot(np.arange(len(Y), len(Y) + 11), forecast, label='Прогноз')
plt.title('Параметрическая NARMA модель и прогноз')
plt.xlabel('Временные шаги')
plt.ylabel('Значения')
plt.legend()
plt.show()
