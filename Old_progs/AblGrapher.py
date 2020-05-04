import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os.path

lis = [
    r'Приведенные реальные ТВЭЛы/00203C4V_250_300.csv',
    r'Приведенные реальные ТВЭЛы/00282C4V_250_300.csv',
    r'Приведенные реальные ТВЭЛы/00359C4V1_250_300.csv',
    r'Приведенные реальные ТВЭЛы/00378C4V_250_300.csv',
    r'Приведенные реальные ТВЭЛы/00381C4V1_250_300.csv',
    r'Приведенные реальные ТВЭЛы/00397C4V1_250_300.csv',
    r'АЧХ для сравнения с реальными ТВЭЛами/УСОП02_250_300.csv',
    r'АЧХ для сравнения с реальными ТВЭЛами/УСОП03_250_300.csv',
    r'АЧХ для сравнения с реальными ТВЭЛами/УСОП05_250_300.csv',
    r'АЧХ для сравнения с реальными ТВЭЛами/УСОП07_250_300.csv',
    r'АЧХ для сравнения с реальными ТВЭЛами/УСОП08_250_300.csv',
    r'АЧХ для сравнения с реальными ТВЭЛами/УСОП09_250_300.csv'
]

lis1 = [
    '00203C4V_250_300',
    '00282C4V_250_300',
    '00359C4V1_250_300',
    '00378C4V_250_300',
    '00381C4V1_250_300',
    '00397C4V1_250_300',
    'УСОП02_250_300',
    'УСОП03_250_300',
    'УСОП05_250_300',
    'УСОП07_250_300',
    'УСОП08_250_300',
    'УСОП09_250_300'
]
num = 0
for lo in lis:
#   загрузка данных из имени файла
    name, ext = os.path.splitext(lo)
    name1 = re.split('/', name)
    name2 = re.split('_', name1[1])
#   загрузка данных в датафрейм
    data = pd.read_csv('{}'.format(lo), header=None) #, decimal=",", delimiter=r"\s+", nrows=1)
#   утверждение границ получаемых частот
    d1 = int(len(data.loc[0]))
    minim = float(name2[1]) # 250 #int(input('ведите минимум диапазона, кГц'))
    maxim = float(name2[2]) # 300 #int(input('ведите максимум диапазона, кГц'))
    d2 = round((maxim-minim)/d1, 7)
#   ниже идет заполнение строк данными о частотах из диапазона частот выше
    data.loc[1] = [minim + i if i == 0 else minim + d2*i for i in range(d1)]
#   находим максимумы амплитуд и частот и записываем их в датафрейм
    data.loc[2, slice(1, len(data.loc[0])-2)] = [data.at[0, i] if data.loc[0, i-1] < data.loc[0, i] > data.loc[0, i+1]
                                                 else 0 for i in range(1,d1-1)]
    data.loc[3, slice(1, len(data.loc[0])-2)] = [data.at[1, i] if data.loc[0, i-1] < data.loc[0, i] > data.loc[0, i+1]
                                                 else 0 for i in range(1,d1-1)]
#   заполняем пустые значения нулями и сохраняем
    data = data.fillna(0)
#   получаем список со значениями по которым мы будем строить регрессионную модель
    amp = [data.loc[0, i-2:i+2] for i in range(2, len(data.loc[0])-2) if data.at[3, i] != 0]
    freq = [data.loc[1, i-2:i+2] for i in range(2, len(data.loc[0])-2) if data.at[3, i] != 0]
#   Цикл для нахождения всех значений добротности для всего диапазона
    m0 = []  # в этом списке находятся все дельта эф подобранные регрессией используемые при расчете декремента
    Q0 = []  # в этом списке находятся все декременты затухания подобранные регрессией
    freq0 = []  # в этом списке находятся все частоты максимумов на которых расчитаны декременты
    for i in range(len(freq)-1):
        Z = np.polyfit(freq[i], amp[i], 2)
        if (float(amp[i][2:-2])/Z[0]) < 0 :
            df = math.sqrt(-float(amp[i][2:-2])/Z[0])
            Q = df/max(freq[i])
            m0.append(df)
            Q0.append(Q)
            freq0.append(max(freq[i]))
        elif (float(amp[i][2:-2])/Z[0]) >=0:
            df = math.sqrt(abs(float(amp[i][2:-2])/Z[0]))
            Q = df/max(freq[i])
            m0.append(df)
            Q0.append(Q)
            freq0.append(max(freq[i]))
#    print(Q0)
    print(round(float(np.mean(Q0)), 6))
#    plt.plot(freq0, Q0, label='{}'.format(lis1[num]))
    num += 1
#    print(Q0)
#    print(freq0)
#    print(round(float(np.std(Q0)), 6))
#    QUO.append(float(np.mean(Q0)))



#    plt.plot(data.loc[1], data.loc[0])
# plt.plot(freq0, Q0)
# plt.grid()
# plt.title('АЧХ СОПа, регрессия +/- 2 точки')   # "Зависимость декремента затухания от частоты резонанса(кГц)")
# plt.legend()
# plt.xlabel('Значение частоты, кГц')
# plt.ylabel('Значение амплитуда')    # декремента затухания(+/- 2 точек)')
# plt.show()
