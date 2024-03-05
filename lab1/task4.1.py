import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Загрузка данных (пример)
data = pd.read_csv("AirQualityUCI.csv", delimiter=';')
data['CO(GT)'] = data['CO(GT)'].replace(',', '.', regex=True).astype(float)

# Выбор значений p и q
p, q = 1, 1

# Создание и обучение модели ARIMA с другими параметрами
model = ARIMA(data['CO(GT)'], order=(p, 0, q))
# results = model.fit()
# Используем другие методы оптимизации и выводим информацию
results = model.fit()
print(results.summary())


# Предсказание n значений вперед
n = 1000
forecast = results.predict(start=len(data), end=len(data) + n - 1, dynamic=False)

# Построение графика
plt.figure(figsize=(40, 5))
plt.plot(data['CO(GT)'], label='Actual Data')
plt.plot(range(len(data), len(data) + n), forecast, label='ARIMA Forecast', linestyle='--', color='red')
plt.title('ARIMA Model Forecast')
plt.xlabel('Time')
plt.ylabel('CO(GT)')
plt.legend()
plt.show()