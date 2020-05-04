import pandas as pd
import numpy as np

lis = [
    r'БНУсредненные массивы рельефности/TVEL_mean_444807-2_97.32.csv',
    r'БНУсредненные массивы рельефности/TVEL_mean_444843-2_98.11.csv',
    r'БНУсредненные массивы рельефности/TVEL_mean_445798-2_98.6.csv',
    r'БНУсредненные массивы рельефности/TVEL_mean_445799-2_98.35.csv'
    ]

"""
lis = [
    r'Чистота гелия/relefnost_00203C4V_96.2_1.csv',
    r'Чистота гелия/relefnost_00282C4V_96.7_11.csv',
    r'Чистота гелия/relefnost_00359C4V1_96.9_7.csv',
    r'Чистота гелия/relefnost_00378C4V_96.6_12.csv',
    r'Чистота гелия/relefnost_00381C4V1_96.8_10.csv',
    r'Чистота гелия/relefnost_00397C4V1_95.8_6.csv'
]
# для чистоты гелия
lis = [
    r'Усредненные массивы рельефности/SOP_mean_2_5.csv',
    r'Усредненные массивы рельефности/SOP_mean_3_8.csv',
    r'Усредненные массивы рельефности/SOP_mean_5_6.csv',
    r'Усредненные массивы рельефности/SOP_mean_7_6.csv',
    r'Усредненные массивы рельефности/SOP_mean_8_6.csv',
]
"""
# Функция которая высчитывает коэффициент детерминации полинома 1й степени
def get_r2_1(x, y):
    zx = (x - np.mean(x)) / np.std(x, ddof=1)
    zy = (y - np.mean(y)) / np.std(y, ddof=1)
    r = np.sum(zx * zy) / (len(x) - 1)
    return r ** 2

# Функция которая высчитывает коэффициент детерминации полинома 2й степени
def get_r2_2(x, y):
    zx = (x - np.mean(x)) / np.std(x, ddof=2)
    zy = (y - np.mean(y)) / np.std(y, ddof=2)
    r = np.sum(zx * zy) / (len(x) - 1)
    return r ** 2

def gettable():
    global lis, data
    table = []
    for q in lis:
        data0 = pd.read_csv(q, header=None)
        j = len(data0.loc[0])
        for i in range(len(data0)):
            print(i)
            if float(data0.at[i, 1]) != 0:
                data0.at[i, j] = (float(data0.at[i, 5]) / float(data0.at[i, 1]))
            else:
                data0.at[i, j] = 99
        table.append(data0)
    data = pd.concat(table, axis=1, ignore_index=True)
    print(data)
    print(data.loc[0:6, 0:6])
    data = data.fillna(0)


def getinfo():
    global data
    list0 = []
    for i in range(len(data)):
        # y = data.loc[i, 1::6].tolist()   # рельефность для измерений где нет параметра СКО
        # x = data.loc[i, 2::6].tolist()   # чистота гелия\давление для измерений где нет параметра СКО
        y = data.loc[i, 1::7].tolist()  # рельефность для измерений где есть параметр СКО
        x = data.loc[i, 2::7].tolist()  # чистота гелия\давление для измерений где есть параметр СКО
        print(y)
        print(x)
        z = np.polyfit(x, y, 1)
        z1 = np.polyfit(x, y, 2)
        r2_1 = get_r2_1(x, y)
        r2_2 = get_r2_2(x, y)
        z0 = np.mean(data.loc[i, 6::7].tolist())   # если есть несколько измерений
        z00 = np.std(data.loc[i, 6::7].tolist())   # если есть несколько измерений
        z = list(z)
        z1 = list(z1)
        list0.append((z, z1, r2_1, r2_2, z0, z00))  # если есть несколько измерений для каждого объекта
        # list0.append((z, z1, r2_1, r2_2))  # если есть по одному измерению для каждого объекта
        print(i)
        print((z, z1, r2_1, r2_2, z0, z00))    # если есть несколько измерений для каждого объекта
        # print((z, z1, r2_1, r2_2))  # если есть по одному измерению для каждого объекта
    data1 = pd.DataFrame(list0)
    print(data1)
    data = pd.concat([data, data1], axis=1, ignore_index=True)
    print(data)
    data.to_csv('final_relefnost_TVEL_BN.csv', header=False, index=False)    # для ТВЭЛов БН
    # data.to_csv('final_relefnost_SOP.csv', header=False, index=False)    # для СОПов где есть СКО
    # data.to_csv('final_relefnost_TVEL.csv', header=False, index=False)    # для твэлов где нет СКО

gettable()

getinfo()