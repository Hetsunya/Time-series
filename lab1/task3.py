import pandas as pd
import matplotlib.pyplot as plt

# 1

# Загрузка данных и предварительная обработка (заменяем запятые на точки и преобразуем в числа)
data = pd.read_csv("AirQualityUCI.csv", delimiter=';')
numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
data[numeric_columns] = data[numeric_columns].replace(',', '.', regex=True).astype(float)

# Аппроксимация функции скользящим средним с различными окнами
window_sizes = [3, 5, 10, 50]

plt.figure(figsize=(20, 8))
plt.plot(data['CO(GT)'], label='Original Data', alpha=0.8)

for window_size in window_sizes:
    rolling_mean = data['CO(GT)'].rolling(window=window_size).mean()
    plt.plot(rolling_mean, label=f'Rolling Mean (Window {window_size})')

plt.title('Moving Average Approximation')
plt.xlabel('Time')
plt.ylabel('CO(GT)')
plt.legend()
plt.show()

# 2
from statsmodels.tsa.arima.model import ARIMA

# Выберем окно для скользящего среднего, например, окно 5
window_size = 10
rolling_mean = data['CO(GT)'].rolling(window=window_size).mean()

# Удалим пропущенные значения, которые могут возникнуть из-за скользящего среднего
rolling_mean = rolling_mean.dropna()

# Создание и обучение модели MA
model = ARIMA(data['CO(GT)'], order=(0, 0, window_size-1))
results = model.fit()

# Построение графика предсказания
plt.figure(figsize=(20, 8))
plt.plot(data['CO(GT)'], label='Original Data', alpha=0.8)
plt.plot(rolling_mean, label=f'Rolling Mean (Window {window_size})', linestyle='--', alpha=0.8)
plt.plot(results.predict(start=window_size, end=len(data)-1), label='MA Model Prediction', linestyle='--', alpha=0.8)

plt.title('Moving Average Model Prediction')
plt.xlabel('Time')
plt.ylabel('CO(GT)')
plt.legend()
plt.show()

