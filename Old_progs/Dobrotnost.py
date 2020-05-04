import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', None)
import pandas as pd
import numpy as np
import math

#

data1 = pd.read_csv(r'../Усредненные АЧХ/соп09_usredn.csv', header=None)

d1 = int(len(data1.loc[0]))
shag = 5 #int(input(r'введите шаг. кГц/c'))
minim = 100 #int(input('ведите минимум диапазона, кГц'))
maxim = 500 #int(input('ведите максимум диапазона, кГц'))
d2 = round((maxim-minim)/d1, 7)
# ниже идет заполнение строк данными о частотах из диапазона частот выше
for i in range(d1):
    if i == 0:
        data1.at[1, i] = (minim + i)
        i += 1
    else:
        data1.at[1, i] = minim + d2*i
        i += 1
# иже идет заполнение строк порядковыми номерами
for w in range(d1):
    data1.at[2, w] = w
# поиск индексов в массиве, где происходит резонанс
for w in range(d1-2): # резонанс
    if data1.at[0, w] < data1.at[0, (w + 1)] > data1.at[0, (w + 2)]:
        data1.at[3, w + 1] = w + 1
    else:
        data1.at[3, w + 1] = 0
for w in range(d1-2): # антирезонанс
    if data1.at[0, w] > data1.at[0, (w + 1)] < data1.at[0, (w + 2)]:
        data1.at[3, w + 1] = w + 1
    else:
        data1.at[3, w + 1] = 0

data1.at[3, 0] = 0
data1.at[3, d1-1] = 0
i = 0
for i in range(d1):
    data1.at[4,i] = 0
    data1.at[5,i] = 0
# ниже расчет отклонений влево от частоты резонанса
for i in range(d1):
    if data1.at[3, i] != 0:
        for j in range(55): #55 для усредненных значений и 25 для не усредненных
            if i-j < 1:
                break
            else:
                if float(data1.at[0, i]) * 0.7 * 0.99 <= float(data1.at[0, i-j]) < float(data1.at[0, i]) * 0.7 * 1.01:
                    data1.at[4, i-1] = float(data1.at[1, i-j])
                    break
                else:
                    continue
    else:
        continue
# ниже расчет отклонений вправо от частоты резонанса
for i in range(d1):
    if data1.at[3, i] != 0:
        for j in range(55): #55 для усредненных значений и 25 для не усредненных
            if i+j >= d1:
                break
            else:
                if float(data1.at[0, i]) * 1.41 * 0.99 <= float(data1.at[0, i+j]) < float(data1.at[0, i]) * 1.41 * 1.01:
                    data1.at[4, i+1] = float(data1.at[1, i+j])
                    break
                else:
                    continue
    else:
        continue

# ниже расчет добротностей и усредненной добротности
y0 = []
dobr = []
for k in range(d1):
    if data1.loc[3, k] != 0 and data1.loc[4, k-1] != 0 and data1.loc[4, k+1] != 0 and data1.loc[4, k+1] - data1.loc[4, k-1] != 0:
        dobr.append((math.pi*data1.loc[1, k]/((((data1.loc[4, k+1] - data1.loc[4, k-1])/2)**2)**(1/2)))/1000)
        y0.append(data1.loc[1, k])
    elif data1.loc[3, k] != 0 and data1.loc[4, k-1] != 0 and data1.loc[4, k+1] == 0:
        dobr.append((math.pi*data1.loc[1, k]/(((data1.loc[1, k] - data1.loc[4, k-1])**2)**(1/2)))/1000)
        y0.append(data1.loc[1, k])
    elif data1.loc[3, k] != 0 and data1.loc[4, k-1] == 0 and data1.loc[4, k+1] != 0:
        dobr.append((math.pi*data1.loc[1, k]/(((data1.loc[1, k] - data1.loc[4, k+1])**2)**(1/2)))/1000)
        y0.append(data1.loc[1, k])
    else:
        k += 1
# Фильтр добротности запись ниже необходима для исключения сильно
# выпадающих значений добротности, чтобы уменьшить ее стандартное отклонение
print(dobr)
print( 'Значений добротности было ' + str(len(dobr)))
z = (len(dobr))
"""
while (np.std(dobr)/np.mean(dobr)) > 0.39:  # для маленьких диапазонов можно использовать 0.39/49 для больших не меньше 0.55
    x = len(dobr)
    i = 2
    while i < x:
        if 0.49 < (np.mean(dobr[:i-1]) - dobr[i])**2:  # должно быть 0.36, для больших значений будет 0.25
            dobr.pop(i)
            y0.pop(i)
            x = len(dobr)
            i += 1
        else:
            i += 1

"""
print( 'Значений добротности стало ' + str(len(dobr))+ ' которое составило ' + str(round((len(dobr))/z, 2)))

a = float(np.mean(dobr))
b = float(np.std(dobr))
print('Среднее значение добротности ' + str(round(a, 4)) + ' со стандартным отклонением ' + str(round(b, 4)))
sop = pd.DataFrame([dobr, y0])

sop.to_csv('СОП09_добротность.csv', index=False, header=False)
"""
fig = plt.figure()
plt.plot(y0, dobr, color='blue', label = 'Добротность СОП-3')
plt.plot(data1.loc[1], data1.loc[0], color='yellow')
plt.legend()
plt.show()


for i in range(d1):
    if data1.at[3, i] != 0:
        if i+j > d1-1:
            break
        else:
            for j in range(40):
                if float(data1.at[0, i]) * 0.7 * 0.95 <= float(data1.at[0, i+j]) <= float(data1.at[0, i]) * 0.7 * 1.05:
                    data1.at[5, i+1] = float(data1.at[1, i+j])
                    break
                else:
                    continue
    else:
        continue

"""