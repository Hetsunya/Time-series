import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('EuStockMarkets.csv')

# Вопрос 1: Существуют ли в наборе данных взаимосвязанные столбцы?
correlation_matrix = df.corr()
print("Матрица корреляции:")
print(correlation_matrix)

# Вопрос 2: Среднее значение и дисперсия
mean_values = df.mean()
variance_values = df.var()
print("\nСреднее значение:")
print(mean_values)
print("\nДисперсия:")
print(variance_values)

# Вопрос 3: Изменяется ли диапазон доступных значений?
# Можно рассмотреть изменение максимального и минимального значения для каждого столбца в течение времени.
max_values = df.max()
min_values = df.min()
range_values = max_values - min_values
print("\nМаксимальные значения:")
print(max_values)
print("\nМинимальные значения:")
print(min_values)
print("\nИзменение диапазона значений:")
print(range_values)

# Вопрос 4: Однородны ли данные?
# Можно рассмотреть изменение стандартного отклонения для каждого столбца в течение времени.
std_deviation_values = df.std()
print("\nСтандартное отклонение:")
print(std_deviation_values)

# Вопрос 5: Построить гистограмму абсолютных значений и гистограмму разностей
absolute_histogram = df.abs().hist(bins=20, figsize=(10, 6))
plt.suptitle('Гистограмма абсолютных значений')
plt.show()

difference_histogram = df.diff().hist(bins=20, figsize=(10, 6))
plt.suptitle('Гистограмма разностей')
plt.show()

# Вопрос 6: Построить две диаграммы рассеяния
plt.scatter(df['DAX'], df['SMI'])
plt.title('Диаграмма рассеяния между DAX и SMI')
plt.xlabel('DAX')
plt.ylabel('SMI')
plt.show()

plt.scatter(df.index, df['DAX'], label='DAX')
plt.scatter(df.index, df['SMI'], label='SMI')
plt.title('Диаграмма рассеяния временных изменений DAX и SMI')
plt.xlabel('Время')
plt.ylabel('Значения')
plt.legend()
plt.show()

# Вопрос 7: Ковариация и ковариационная матрица
covariance = df[['DAX', 'SMI']].cov().iloc[0, 1]
covariance_matrix = df.cov()
print("\nКовариация DAX и SMI:")
print(covariance)
print("\nКовариационная матрица:")
print(covariance_matrix)