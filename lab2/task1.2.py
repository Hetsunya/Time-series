import matplotlib.pyplot as plt
import numpy as np

def step_signal(x, a, b):
    return np.where((x >= a) & (x < b), 1, 0)

# Создаем ступенчатые сигналы
x = np.linspace(-2, 5, 500)
f_signal = step_signal(x, 0, 1)
g_signal = step_signal(x, 1, 2)

# Свертка с использованием NumPy
conv_result = np.convolve(f_signal, g_signal, mode='full')[:len(x)]

# Построение графиков
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, f_signal, label='f(x)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, g_signal, label='g(x)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, conv_result, label='(f * g)(x)')
plt.legend()

plt.tight_layout()
plt.show()
