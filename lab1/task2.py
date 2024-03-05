import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf
from sklearn.metrics import mean_squared_error

# Чтение данных из файла (замените "your_file.csv" на фактическое имя вашего файла)
data = pd.read_csv("AirQualityUCI.csv", delimiter=';')

# Преобразование формата даты и времени
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')

# Замена запятых на точки и преобразование числовых значений в числа с плавающей точкой
numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
data[numeric_columns] = data[numeric_columns].replace(',', '.', regex=True).astype(float)

# Установка столбца DateTime в качестве индекса
data.set_index('DateTime', inplace=True)

# Построение временных рядов
plt.figure(figsize=(15, 10))
for column in numeric_columns:
    plt.plot(data.index, data[column], label=column)

plt.title('Временные ряды переменных')
plt.xlabel('Дата и время')
plt.ylabel('Значение')
plt.legend(loc='upper right')
plt.show()


# Построение графика PACF
plt.figure(figsize=(10, 5))
plot_pacf(data['CO(GT)'], lags=30, title='Partial Autocorrelation Function (PACF) for CO(GT)')
plt.show()

# Преобразование формата даты и времени, хз почему он забывает, но ошибка без этого
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')

# Установка столбца DateTime в качестве индекса
data.set_index('DateTime', inplace=True)

# Определение лага с помощью графика частичной автокорреляционной функции
lag_pacf = pacf(data['CO(GT)'].dropna(), nlags=30)

# Определение лага на основе значений PACF
lag = np.argmax(np.abs(lag_pacf) < 0.05)

# Создание и обучение модели AR
model = AutoReg(data['CO(GT)'].dropna(), lags=1)
results = model.fit()

# Построение графика
plt.figure(figsize=(15, 8))
plt.plot(data['CO(GT)'], label='Actual')
plt.plot(results.fittedvalues, color='red', label='AR Model')
plt.title('AR Model for CO(GT)')
plt.xlabel('DateTime')
plt.ylabel('CO(GT)')
plt.legend()
plt.show()