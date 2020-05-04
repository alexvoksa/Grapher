import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('final_relefnost_TVEL.csv', header=None)

data = data.dropna(axis=0)
data.index = [i for i in range(len(data))]
data = data.drop(np.where(data[0] == 0)[0].tolist())
data.index = [i for i in range(len(data))]

i1 = int(32) # для чистоты гелия
i2 = int(33) # для чистоты гелия

x1 = int(np.where(data[i1] == max(data[i1]))[0])
x2 = int(np.where(data[i2] == max(data[i2]))[0])


if data.at[x1, i1] > data.at[x2, i2]:
    print('Зависимость прямая')
elif data.at[x1, i1] < data.at[x2, i2]:
    print('Зависимость параболическая')
elif data.at[x1, i1] == data.at[x2, i2]:
    print('Тип зависимости не определен')

# xm = max(x1, x2)

xm = list(set(np.where(data[3] == 275.4894)[0].tolist()) & set(np.where(data[4] == 259.8733)[0].tolist()))[0]

y = data.loc[xm, 1:(i1-3):5].to_list()  # для чистоты гелия
x = data.loc[xm, 2:(i1-3):5].to_list()    # для чистоты гелия

deter = data.loc[xm, i1]
par1 = data.loc[xm, i1-2]


k1 = float(data.loc[xm, i1-2].split(',')[0].replace('[', '').replace(' ', ''))  # для давления гелия
k2 = float(data.loc[xm, i1-2].split(',')[1].replace(']', '').replace(' ', ''))


print('Зависимость рельефности от чистоты гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
            str(data.at[xm, 4]) + 'кГц')
print('К-т детерминации: ', deter)
print('Параметны прямой: ', par1)



plt.plot(x, [x*(k1) + k2 for x in x], color='yellow')
plt.scatter(x, y)
plt.xlabel('Чистота гелия, проценты') # для чистоты гелия
plt.ylabel('Рельефность,')
plt.title('График зависимости рельефности от чистоты гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
          str(data.at[xm, 4]) + 'кГц')
plt.show()