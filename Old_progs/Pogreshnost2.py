import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x03 = pd.DataFrame(pd.read_csv(r'../Остальные данные/СОП03 массив.csv'))
x05 = pd.DataFrame(pd.read_csv(r'../Остальные данные/СОП05 массив.csv'))
x08 = pd.DataFrame(pd.read_csv(r'../Остальные данные/СОП08 массив.csv'))
x09 = pd.DataFrame(pd.read_csv(r'../Остальные данные/СОП09 массив.csv'))

d1 = int(len(x03.loc[0]))

shag = 5  # int(input(r'введите шаг. кГц/c'))
minim = 100  # int(input('ведите минимум диапазона, кГц'))
maxim = 500  # int(input('ведите максимум диапазона, кГц'))
d2 = round((maxim-minim)/d1, 7)

y1 = []
for i in range(0, d1, 1):
    if i == 0:
        y1.append(minim + i)
        i += 1
    else:
        y1.append(minim + d2*i)
        i += 1

sko03 = []
sko05 = []
sko08 = []
sko09 = []

mea03 = []
mea05 = []
mea08 = []
mea09 = []

otnos03 = []
otnos05 = []
otnos08 = []
otnos09 = []

for i in range(0, d1, 1):
    sko03.append(np.std(x03, axis=0)[i])
    mea03.append(np.mean(x03, axis=0)[i])
for i in range(0, d1, 1):
    sko05.append(np.std(x05, axis=0)[i])
    mea05.append(np.mean(x05, axis=0)[i])
for i in range(0, d1, 1):
    sko08.append(np.std(x08, axis=0)[i])
    mea08.append(np.mean(x08, axis=0)[i])
for i in range(0, d1, 1):
    sko09.append(np.std(x09, axis=0)[i])
    mea09.append(np.mean(x09, axis=0)[i])

for i in range(0, d1, 1):
    otnos03.append(sko03[i] / mea03[i])
    otnos05.append(sko05[i] / mea05[i])
    otnos08.append(sko08[i] / mea08[i])
    otnos09.append(sko09[i] / mea09[i])

p03 = str(round(float(np.mean(otnos03)), 2)*100)
p05 = str(round(float(np.mean(otnos05)), 2)*100)
p08 = str(round(float(np.mean(otnos08)), 2)*100)
p09 = str(round(float(np.mean(otnos09)), 2)*100)

print('Относительная погрешность измерения СОП03 составила ' + p03 + '%')
print('Относительная погрешность измерения СОП05 составила ' + p05 + '%')
print('Относительная погрешность измерения СОП08 составила ' + p08 + '%')
print('Относительная погрешность измерения СОП09 составила ' + p09 + '%')

fig = plt.figure()
print(fig.axes)
print(type(fig))
plt.plot(y1, otnos03, color='black', label='ОТН.ПОГ СОП03')
plt.plot(y1, otnos05, color='blue', label='ОТН.ПОГ СКО05')
plt.plot(y1, otnos08, color='red', label='ОТН.ПОГ 08')
plt.plot(y1, otnos09, color='yellow', label='ОТН.ПОГ 09')
plt.legend()
plt.show()
print(fig.axes)