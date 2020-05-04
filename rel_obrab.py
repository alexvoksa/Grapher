import pandas as pd
import numpy as np
import re
import os.path
import os
"""
lis = [
    r'Подборка рельефности/relefnost_2_5_1.csv',
    r'Подборка рельефности/relefnost_2_5_2.csv',
    r'Подборка рельефности/relefnost_2_5_3.csv',
    r'Подборка рельефности/relefnost_2_5_4.csv',
    r'Подборка рельефности/relefnost_2_5_5.csv',
    r'Подборка рельефности/relefnost_2_5_6.csv',
    r'Подборка рельефности/relefnost_2_5_7.csv',
    r'Подборка рельефности/relefnost_2_5_8.csv',
    r'Подборка рельефности/relefnost_2_5_9.csv',
    r'Подборка рельефности/relefnost_2_5_10.csv',
    r'Подборка рельефности/relefnost_3_8_1.csv',
    r'Подборка рельефности/relefnost_3_8_2.csv',
    r'Подборка рельефности/relefnost_3_8_3.csv',
    r'Подборка рельефности/relefnost_3_8_4.csv',
    r'Подборка рельефности/relefnost_3_8_5.csv',
    r'Подборка рельефности/relefnost_3_8_6.csv',
    r'Подборка рельефности/relefnost_3_8_7.csv',
    r'Подборка рельефности/relefnost_5_6_8.csv',
    r'Подборка рельефности/relefnost_5_6_9.csv',
    r'Подборка рельефности/relefnost_5_6_10.csv',
    r'Подборка рельефности/relefnost_5_6_11.csv',
    r'Подборка рельефности/relefnost_5_6_12.csv',
    r'Подборка рельефности/relefnost_5_6_13.csv',
    r'Подборка рельефности/relefnost_5_6_14.csv',
    r'Подборка рельефности/relefnost_5_6_15.csv',
    r'Подборка рельефности/relefnost_5_6_16.csv',
    r'Подборка рельефности/relefnost_5_6_17.csv',
    r'Подборка рельефности/relefnost_5_6_18.csv',
    r'Подборка рельефности/relefnost_7_6_11.csv',
    r'Подборка рельефности/relefnost_7_6_12.csv',
    r'Подборка рельефности/relefnost_7_6_13.csv',
    r'Подборка рельефности/relefnost_7_6_14.csv',
    r'Подборка рельефности/relefnost_7_6_15.csv',
    r'Подборка рельефности/relefnost_7_6_16.csv',
    r'Подборка рельефности/relefnost_7_6_17.csv',
    r'Подборка рельефности/relefnost_7_6_18.csv',
    r'Подборка рельефности/relefnost_7_6_19.csv',
    r'Подборка рельефности/relefnost_7_6_20.csv',
    r'Подборка рельефности/relefnost_8_6_19.csv',
    r'Подборка рельефности/relefnost_8_6_20.csv',
    r'Подборка рельефности/relefnost_8_6_21.csv',
    r'Подборка рельефности/relefnost_8_6_22.csv',
    r'Подборка рельефности/relefnost_8_6_23.csv',
    r'Подборка рельефности/relefnost_8_6_24.csv',
    r'Подборка рельефности/relefnost_8_6_25.csv',
    r'Подборка рельефности/relefnost_8_6_26.csv',
    r'Подборка рельефности/relefnost_8_6_27.csv',
    r'Подборка рельефности/relefnost_8_6_28.csv'
]
"""

dir = 'БНЧистота гелия'
lis = os.listdir(dir)


def getlist2():
    global lis, list2
    list2 = []
    for link in lis:
        name, ext = os.path.splitext(link)
        name1 = re.split('/', name)
        name2 = re.split('_', name1[1])
        list2.append(name2[1])
    list2 = list(set(list2))


def checkin():
    global k, lis, mainframe, name2
    table = []
    for link in lis:
        name, ext = os.path.splitext(link)
        name1 = re.split('/', name)
        name2 = re.split('_', name1[1])
        if name2[1] == k:
            part = pd.read_csv(link, header=None)
            table.append(part)
    if len(table) != 0:
        mainframe = pd.concat(table, axis=1, ignore_index=True)
    print(mainframe)


def getframe():
    global mainframe, name2
    result = []
    for i in range(len(mainframe)):
        print(i)
        a = np.mean(mainframe.loc[i, 1::5])
        b = np.std(mainframe.loc[i, 1::5])
        result.append((mainframe.at[i, 0], a, mainframe.at[i, 2], mainframe.at[i, 3], mainframe.at[i, 4], b))
        print(a, b)
    data = pd.DataFrame(result)
    print(data)
    data.to_csv(r'БНУсредненные массивы рельефности/TVEL_mean_{}_{}.csv'.format(mainframe.at[0, 0],
                                                                             mainframe.at[0, 2]), header=False,
                index=False)

getlist2()

for k in list2:
    checkin()
    getframe()
