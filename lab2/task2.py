import numpy as np
import matplotlib.pyplot as plt


# Определение выбранных функций
#915
def f1(x):
    return np.sin(np.abs(2 * x))  # Функция 1: y = sin(|2x|)
#929
def f2(x):
    return np.abs(x) - np.sin(x)  # Функция 2: y = |x| - sin(x)
 #865
def f3(x):
    return np.log(np.abs(x - 1))  # Функция 3: y = ln(|x - 1|)
#831
def f4(x):
    return 4 * x**2 * np.exp(2 * x)  # Функция 4: y = 4x^2 * e^(2x)
#667
def f5(x):
    return np.sqrt(np.abs(x) + 1)  # Функция 5: y = sqrt(|x| + 1)



# Функция для вычисления взаимнокорреляционной функции
def cross_correlation(x, y, max_lag):
    result = np.zeros(2 * max_lag + 1, dtype=float)

    for lag in range(-max_lag, max_lag + 1):
        sum_val = 0
        for i in range(len(x)):
            j = i - lag
            if 0 <= j < len(y):
                sum_val += x[i] * y[j]
        result[lag + max_lag] = sum_val

    return result


# Создание массива значений x
x_values = np.linspace(-5, 5, 1000)

# Построение графиков для каждой пары функций
functions = [f1, f2, f3, f4, f5]

for i in range(len(functions)):
    for j in range(i + 1, len(functions)):
        # Вычисление значений функций
        y1 = functions[i](x_values)
        y2 = functions[j](x_values)

        # Вычисление взаимнокорреляционной функции
        cross_corr_result = cross_correlation(y1, y2, 50)
        lags = np.arange(-50, 51)

        # Построение графиков
        plt.figure(figsize=(10, 6))

        plt.subplot(3, 1, 1)
        plt.plot(x_values, y1, label=f'Function {i + 1}')
        plt.plot(x_values, y2, label=f'Function {j + 1}')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(lags, cross_corr_result, label='Cross-correlation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.stem(lags, cross_corr_result, label='Discrete Cross-correlation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.legend()

        plt.tight_layout()
        plt.show()
