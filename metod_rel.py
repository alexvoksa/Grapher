import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os.path

#  здесь будет список файлов для оценки

names = [
    r'Все без проволки/00203C4V_10_400_1,0_1.txt',
    r'Все без проволки/00282C4V_10_400_1,0_11.txt',
    r'Все без проволки/00359C4V1_10_400_1,0_7.txt',
    r'Все без проволки/00378C4V_10_400_1,0_12.txt',
    r'Все без проволки/00381C4V1_10_400_1,0_10.txt',
    r'Все без проволки/00397C4V1_10_400_1,0_6.txt',
    r'Все без проволки/00416B4_10_400_1,0_11.txt',
    r'Все без проволки/00416B4V1_10_400_1,0_5.txt',
    r'Все без проволки/00417B4V_10_400_1,0_2.txt',
    r'Все без проволки/00521B4_10_400_1,0_15.txt',
    r'Все без проволки/00521B4V1_10_400_1,0_4.txt',
    r'Все без проволки/00521B4V_10_400_1,0_3.txt',
    r'Все без проволки/00538B4V1_10_400_0,5_9.txt',
    r'Все без проволки/00538B4V1_10_400_1,0_8.txt'
]
'''
names = [
    r'БН/444807-2_10_400_1,0_1.txt',
    r'БН/444807-2_10_400_1,0_2.txt',
    r'БН/444807-2_10_400_1,0_3.txt',
    r'БН/444843-2_10_400_1,0_1.txt',
    r'БН/444843-2_10_400_1,0_2.txt',
    r'БН/444843-2_10_400_1,0_3.txt',
    r'БН/444843-2_10_400_1,0_4.txt',
    r'БН/445798-2_10_400_1,0_1.txt',
    r'БН/445798-2_10_400_1,0_2.txt',
    r'БН/445798-2_10_400_1,0_3.txt',
    r'БН/445799-2_10_400_1,0_1.txt',
    r'БН/445799-2_10_400_1,0_2.txt',
    r'БН/твэл02-2_10_400_1,0_1.txt',
    r'БН/твэл03-2_10_400_1,0_2.txt'
]
'''
dataz = pd.read_csv('data_kalibr_MOX_0.csv')
minfre = dataz.at[0, 'minint']
maxfre = dataz.at[0, 'maxint']
k1 = dataz.at[0, 'k1']
k2 = dataz.at[0, 'k2']
num_num = 0


def filter():
    global data
    global maxfre
    global minfre
    ampl = []
    freq = []
    for i in range(len(data.loc[1])):
        if maxfre >= data.at[1, i] >= minfre:
            ampl.append(data.at[0, i])
            freq.append(data.at[1, i])
    #   Создаем новый отфильтрованный датафрейм
    data = pd.DataFrame([ampl, freq])


def countrel():
    global k1, k2, data, zz, dataz, num_num
    diff = np.diff(data.loc[0].to_list()) / np.diff(data.loc[1].to_list())
    rel = round(np.trapz(abs(diff), x=data.loc[1, 1:].to_list()) /
                np.trapz(data.loc[0].to_list(), x=data.loc[1].to_list()), 5)
    clear = (rel - k2) / k1
    print(zz, clear, '%')
    print(dataz)
    q = [zz, num_num, 'MOX_BN', data.at[1, 0], data.at[1, len(data.loc[1])-1], 0, 0, 0, rel, clear]
    qq = ['name', 'num', 'type', 'minint', 'maxint', 'k1', 'k2', 'det', 'rel', 'clear']
    cc = pd.DataFrame([q], columns=qq)
    dataz = dataz.append(cc, ignore_index=True)
    dataz.to_csv('data_kalibr_MOX.csv', index=False)
    # dataz['nominal'] = [0, 97.32, 97.32, 97.32, 98.11, 98.11, 98.11, 98.11, 98.6, 98.6, 98.6, 98.35, 98.35] топливо БН
    print(dataz)
    num_num += 1


def getrel():
    global names, k1, k2, dataz, data, zz
    for lo in names:
        name, ext = os.path.splitext(lo)
        name1 = re.split('/', name)
        name2 = re.split('_', name1[1])
        zz = name2[0]+'_'+name2[4]  # используется для чистоты гелия
        #   загрузка данных в датафрейм
        data = pd.read_csv('{}'.format(lo), header=None, decimal=",", delimiter=r"\s+", nrows=1)
        #   утверждение границ получаемых частот
        d1 = int(len(data.loc[0]))
        minim = float(name2[1])
        maxim = float(name2[2])
        d2 = round((maxim - minim) / d1, 7)
        #   ниже идет заполнение строки данными о частотах из диапазона частот выше
        data.loc[1] = [minim + i * d2 for i in range(d1)]
        print(data)
        #   фильтрация в диапазон minfre-maxfre
        filter()
        print(data)
        #   расчет рельефности
        countrel()


#getrel()

dataz['nominal'] = [0, 96.2, 96.7, 96.9, 96.6, 96.8, 95.8]
dataz['err'] = abs(dataz['nominal'] - dataz['clear'])

lis = list(dataz.loc[1:, 'err'] / dataz.loc[1:, 'clear'])
lis.insert(0, 0)
dataz['otn_err'] = lis
dataz['err_2'] = [0, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27]


plt.plot(np.linspace(95.7, 97), list(np.linspace(95.7, 97)*k1+k2), label='Калибровка', alpha=0.4)
for i in range(1, len(dataz)):
    plt.scatter(dataz.loc[i, 'nominal'], dataz.loc[i, 'rel'], label=dataz.loc[i, 'name'])
    plt.errorbar(dataz.loc[i, 'nominal'], dataz.loc[i, 'rel'], xerr=dataz.loc[i, 'err_2'], alpha=0.4, color='red', capsize=10)
#  , yerr=dataz.loc[i, 'err']
plt.legend()
plt.title('График зависимости рельефности от чистоты гелия\nдля ТВЭЛов c MOX-топливом \nдиапазон '
          + str((minfre, maxfre)))
plt.xlabel('Чистота гелия, %')
plt.ylabel('Рельефность')
plt.grid()
plt.show()