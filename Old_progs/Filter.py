import pandas as pd


lis = [
    r'Усредненные АЧХ/соп02_usredn100500.csv',
    r'Усредненные АЧХ/соп03_usredn.csv',
    r'Усредненные АЧХ/соп05_usredn.csv',
    r'Усредненные АЧХ/соп07_usredn100500.csv',
    r'Усредненные АЧХ/соп08_usredn.csv',
    r'Усредненные АЧХ/соп09_usredn.csv'
#    r'Реальные ТВЭЛы/00203C4V_10_400_1,0_1.txt',
#    r'Реальные ТВЭЛы/00282C4V_10_400_1,0_11.txt',
#    r'Реальные ТВЭЛы/00359C4V1_10_400_1,0_7.txt',
#    r'Реальные ТВЭЛы/00378C4V_10_400_1,0_12.txt',
#    r'Реальные ТВЭЛы/00381C4V1_10_400_1,0_10.txt',
#    r'Реальные ТВЭЛы/00397C4V1_10_400_1,0_6.txt'
    ]

lis1 = [#'00203C4V_250_300', '00282C4V_250_300', '00359C4V1_250_300',
        # '00378C4V_250_300', '00381C4V1_250_300', '00397C4V1_250_300']

 'УСОП02_250_300', 'УСОП03_250_300', 'УСОП05_250_300', 'УСОП07_250_300', 'УСОП08_250_300', 'УСОП09_250_300']

num = 0
for lo in lis:
    data1 = pd.read_csv('{}'.format(lo), header=None) #, decimal=",", delimiter=r"\s+", nrows=1)

    mi = 250  # float(input('Введите нужный минимум исследуемого интервала'))
    ma = 300  # float(input('Введите нужный максимум исследуемого интервала'))

    d1 = int(len(data1.loc[0]))
    minim = 100  # int(input('ведите минимум диапазона, кГц'))
    maxim = 500  # int(input('ведите максимум диапазона, кГц'))
    d2 = round((maxim - minim) / d1, 7)

    # ниже идет заполнение строк данными о частотах из диапазона частот выше
    for i in range(d1):
        if i == 0:
            data1.at[1, i] = (minim + i)
            i += 1
        else:
            data1.at[1, i] = minim + d2 * i
            i += 1
    #    print(data1)
    # поиск и замена значений не попадающих в интервал

    for i in range(d1):
        if data1.at[1, i] < mi:
            data1.at[0, i] = 9999
            data1.at[1, i] = 9999

    #    print(data1)
    for i in range(d1):
        if 9999999999 > data1.at[1, i] > ma:
            data1.at[0, i] = 9999999999
            data1.at[1, i] = 9999999999

    #    print(data1)
    x1 = []
    x2 = []
    for i in range(d1):
        if data1.at[0, i] != 9999999999:
            x1.append(data1.at[0, i])
            x2.append(data1.at[1, i])
    # создание нового списка
    data0 = pd.DataFrame([x1, x2])

    print(data0)

    data0.to_csv(r'АЧХ для сравнения с реальными ТВЭЛами/{}.csv'.format(lis1[num]), index=False, header=False)
    num += 1
