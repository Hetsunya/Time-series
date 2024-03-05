import pandas as pd
import matplotlib.pyplot as plt

# Чтение данных из файла
file_path = 'lab3_data6.txt'
data = pd.read_csv(file_path, sep='\t', header=None)

# Визуализация временных рядов
plt.figure(figsize=(10, 6))

for i in range(data.shape[1]):
    plt.plot(data[i], label=f'Тип музыки {i + 1}')

plt.title('Сердечный ритм во время прослушивания разных типов музыки')
plt.xlabel('Время')
plt.ylabel('Интервалы между ударами сердца (мс)')
plt.legend()
plt.show()
