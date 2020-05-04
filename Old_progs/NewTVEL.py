import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lis = [0.0004720, 0.000472, 0.000495, 0.000461, 0.000478, 0.000539]
lis1 = [96.2, 96.7, 96.9, 96.6, 96.8, 95.8]
name = ['00203 Ц4', '00282 Ц4', '00359 Ц4', '00378 Ц4', '00381 Ц4', '00397 Ц4']

data = pd.DataFrame(data=[lis, lis1], columns=name)
data1 = data.T

data1 = data1.sort_values(by=[0])
data1 = data1.sort_values(by=[1])
print(data1)

Z = np.polyfit(data1[0], data1[1], 2)

K = [Z[0]*x**2 + Z[1]*x + Z[2] for x in data1[0]]
plt.plot(data1[1], data1[0])
plt.plot(K, data1[0])


plt.show()