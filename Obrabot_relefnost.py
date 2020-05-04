import pandas as pd
import numpy as np
import re
import os.path
from tqdm import tqdm
import os


def countrel():
    pressure = {
        'СОП-02': 5,
        'СОП-03': 8,
        'СОП-05': 6,
        'СОП-07': 6,
        'СОП-08': 6
    }

    dir = 'АЧХ чистые'
    lis = os.listdir(dir)

    minfre = int(
        100)  # задание переменной которая отвечает за нижн границу диапазона к которому будут приводиться данные
    maxfre = int(
        500)  # задание переменной которая отвечает за верх границу диапазона к которому будут приводиться данные

    def createfinalframe():
        nonlocal minfre, maxfre, pressure

        def filter():
            nonlocal data, maxfre, minfre, pressure, d2
            ampl = []
            freq = []
            print('Фильтрую данные')
#            for i in range(len(data.loc[1])):
#                if maxfre >= data.at[1, i] >= minfre:
#                    ampl.append(data.at[0, i])
#                    freq.append(data.at[1, i])
            #   Создаем новый отфильтрованный датафрейм
            data = data.loc[:, (data.loc[1] >= minfre) & (data.loc[1] <= maxfre)]
            data.columns = [i for i in range(len(data.loc[0]))]

            d1 = int(len(data.loc[0]))
            d2 = round((maxfre - minfre) / d1, 7)
            #   Перезапись частот. Погрешность может составлять 40 Гц!
            data.loc[1] = [minfre + i * d2 for i in range(d1)]
            print('Данные отфильтрованы в диапазон ', str(minfre) + '-' + str(maxfre), 'кГц')

        def relef():
            nonlocal data, zz, name2, pressure, d2
            compare = []

            for k in tqdm(range(2, len(data.loc[0])), desc='Расчитываю рельефности для диапазонов', leave=True):
                m = 0
                for y in range(1, len(data.loc[0]), k):
                    diff = np.diff(data.loc[0, m: y].to_list()) / np.diff(data.loc[1, m: y].to_list())
                    rel = round(np.trapz(abs(diff), x=data.loc[1, m + 1:y].to_list()) /
                                np.trapz(data.loc[0, m:y].to_list(), x=data.loc[1, m:y].to_list()), 5)
                    press = pressure.get(name2[0])
                    intermax = round(float(data.loc[1, y]), 4)
                    intermin = round(float(data.loc[1, m]), 4)
                    spec = (zz, rel, press, intermax, intermin)
                    compare.append(spec)
                    m = y

            megadata = pd.DataFrame(compare)
            print('Сохраняем файл...')
            megadata.to_csv(r'Обработанные АЧХ/relefnost_{}_{}_{}.csv'.format(zz, press, int(name2[4])), header=False,
                            index=False)
            megadata = 0
            print('Файл с данными по {} успешно сохранен!'.format(zz))

        for lo in lis:
            #  загрузка данных из имени файла
            name, ext = os.path.splitext(lo)
            name2 = re.split('_', name)
            print('Загружаю данные АЧХ для:', name2[0] + '_' + name2[4])
            zz = name2[0]
            #   загрузка данных в датафрейм
            path_correct = dir + '/' + lo  # Создание пути файла
            data = pd.read_csv('{}'.format(path_correct), header=None, decimal=",", delimiter=r"\s+", nrows=1)
            #   утверждение границ получаемых частот
            d1 = int(len(data.loc[0]))
            minim = float(name2[1])
            maxim = float(name2[2])
            d2 = round((maxim - minim) / d1, 7)
            #   ниже идет заполнение строк данными о частотах из диапазона частот выше
            data.loc[1] = [minim + i * d2 for i in range(d1)]
            filter()  # фильтрация в диапазон minfre-maxfre
            relef()  # расчет рельефности

    createfinalframe()


def avg_rel():
    global avg


    if avg == 'Y':
        print('Модуль усреднения включен')
        dir = 'Обработанные АЧХ'
        lis = os.listdir(dir)
        list2 = []

        def getlist2():  # Получает данные из списка файлов
            nonlocal lis, list2
            for lo in lis:
                name, ext = os.path.splitext(lo)
                name2 = re.split('_', name)
                list2.append(name2[1])
            list2 = list(set(list2))

        def checkin():
            nonlocal k, lis

            def getframe():
                nonlocal mainframe
                result = []
                for i in tqdm(range(len(mainframe)), desc='Расчет срзнач и ско', leave=True):
                    a = np.mean(mainframe.loc[i, 1::5])
                    b = np.std(mainframe.loc[i, 1::5])
                    result.append(
                        (mainframe.at[i, 0], a, mainframe.at[i, 2], mainframe.at[i, 3], mainframe.at[i, 4], b))
                data = pd.DataFrame(result)
                print(data)
                print('Сохраняем данные по', mainframe.at[0, 0])
                data.to_csv(r'Усредненные массивы данных/SOP_mean_{}_{}.csv'.format(mainframe.at[0, 0],
                                                                                         mainframe.at[0, 2]),
                            header=False,
                            index=False)
                print('Данные по', mainframe.at[0, 0], 'сохранены')

            table = []
            for link in lis:
                name, ext = os.path.splitext(link)
                name2 = re.split('_', name)
                path_correct = dir + '/' + link
                if name2[1] == k:
                    part = pd.read_csv(path_correct, header=None)
                    table.append(part)
            if len(table) != 0:
                mainframe = pd.concat(table, axis=1, ignore_index=True)
            getframe()

        getlist2()

        for k in list2:
            checkin()


def get_final_frame():
    global avg
    if avg == 'Y':
        dir = 'Усредненные массивы данных'
    else:
        dir = 'Обработанные АЧХ'

    lis = os.listdir(dir)

    def gettable():
        nonlocal lis
        global avg

        table = []
        print('Идет объединение данных')

        for lo in lis:
            data0 = pd.read_csv(lo, header=None)
            table.append(data0)
        data = pd.concat(table, axis=1, ignore_index=True)
        data.fillna(0)
        print('Объединение данных успешно завершено')


        def get_r2_1(x, y):
            zx = (x - np.mean(x)) / np.std(x, ddof=1)
            zy = (y - np.mean(y)) / np.std(y, ddof=1)
            r = np.sum(zx * zy) / (len(x) - 1)
            return r ** 2


        def get_r2_2(x, y):
            zx = (x - np.mean(x)) / np.std(x, ddof=2)
            zy = (y - np.mean(y)) / np.std(y, ddof=2)
            r = np.sum(zx * zy) / (len(x) - 1)
            return r ** 2


        list0 = []
        for i in tqdm(range(len(data)), desc='Расчет параметров зависимостей', leave=True):
            if avg == 'Y':  # если есть несколько измерений для каждого объекта и есть СКО
                x = data.loc[i, 2::7].tolist()  # чистота гелия/давление
                y = data.loc[i, 1::7].tolist()  # рельефность
                z = list(np.polyfit(x, y, 1))
                z1 = list(np.polyfit(x, y, 2))
                r2_1 = get_r2_1(x, y)
                r2_2 = get_r2_2(x, y)
                z0 = np.mean(data.loc[i, 5::7].tolist())  # если есть несколько измерений
                z00 = np.std(data.loc[i, 5::7].tolist())  # если есть несколько измерений
                if z0 != 0:
                    z_otn_sko = z00 / z0
                else:
                    z_otn_sko = 99
                list0.append((z[0], z[1], z1[0], z1[1], z1[2], r2_1, r2_2, z0, z00, z_otn_sko))
            else:  # если есть по одному измерению для каждого объекта и нет СКО
                y = data.loc[i, 1::6].tolist()  # рельефность
                x = data.loc[i, 2::6].tolist()  # чистота гелия/давление
                z = np.polyfit(x, y, 1)
                z1 = np.polyfit(x, y, 2)
                r2_1 = get_r2_1(x, y)
                r2_2 = get_r2_2(x, y)
                z = list(z)
                z1 = list(z1)
                list0.append((z, z1, r2_1, r2_2))

        data1 = pd.DataFrame(list0)
        data = pd.concat([data, data1], axis=1, ignore_index=True)

        print('Идет сохранение файла')
        data.to_csv('final_relefnost_SOP.csv', header=False, index=False)
        print('Файл успешно сохранен!')

    gettable()


countrel()  # считаем рельефности для всех диапазонов и сохраняем их в определенную папку
# мне еще надо дописать выбор папки для сохранения и указания пути к файлам при загрузке
# еще желательно добавить графический интерфейс

avg = input('Включить модуль усреднения? Y/N')
avg = str(avg)
avg = avg.upper()
avg.replace('Н', 'Y')
avg.replace('Т', 'N')
if avg == 'Y':
    avg_rel()
get_final_frame()
print('Ну вот и все, не прошло и года')



