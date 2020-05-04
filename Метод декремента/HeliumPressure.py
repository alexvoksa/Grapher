import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os.path

pressure = {
    'СОП-02': 5,
    'СОП-03': 8,
    'СОП-05': 6,
    'СОП-07': 6,
    'СОП-08': 6
}  # в этом словаре находятся названия файлов и их давления или чистота гелия
Y = []
L = []
XYZ = float(0.6)  # в этой переменной записано наибольшее значение СКО интервала для декремента затухания
inter = int(2)  # в этой переменной записан шаг с которым мы проходим по общему диапазону
minscore = 0.5  # в этой переменной записано наименьшее допустимое значение к-та R^2
lis = [
    r'АЧХ чистые/СОП-02_0_700_5,0_1.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_2.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_3.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_4.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_5.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_6.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_7.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_8.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_9.txt',
    r'АЧХ чистые/СОП-02_0_700_5,0_10.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_1.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_2.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_3.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_4.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_5.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_6.txt',
    r'АЧХ чистые/СОП-03_100_500_5,0_7.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_8.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_9.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_10.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_11.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_12.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_13.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_14.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_15.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_16.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_17.txt',
    r'АЧХ чистые/СОП-05_100_500_5,0_18.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_11.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_12.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_13.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_14.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_15.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_16.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_17.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_18.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_19.txt',
    r'АЧХ чистые/СОП-07_0_700_5,0_20.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_19.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_20.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_21.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_22.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_23.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_24.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_25.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_26.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_27.txt',
    r'АЧХ чистые/СОП-08_100_500_5,0_28.txt'
]  # это список с путем к исследоваемым файлам ВНИМАНИЕ путь к файлу должен быть указан строго как здесь

names = []
finalframe = pd.DataFrame()
data0 = pd.DataFrame()

minfre = int(100)  # задание переменной которая отвечает за нижн границу диапазона к которому будут приводиться данные
maxfre = int(400)  # задание переменной которая отвечает за верх границу диапазона к которому будут приводиться данные


# Функция которая загружает данные по всем файлам из списка lis, прописывает для каждого частоты ориентируясь на
# название файла и приводит данные к частотному диапазону указанному в переменных minfre и maxfre
def createfinalframe():
    global finalframe
    global minfre
    global maxfre
    global inter
    num = 0
    means = []
    sko = []
    list13 = []
    freql = []

    for lo in lis:
        #  загрузка данных из имени файла
        name, ext = os.path.splitext(lo)
        name1 = re.split('/', name)
        name2 = re.split('_', name1[1])
        #   загрузка данных в датафрейм
        data = pd.read_csv('{}'.format(lo), header=None, decimal=",", delimiter=r"\s+", nrows=1)
        #   утверждение границ получаемых частот
        d1 = int(len(data.loc[0]))
        minim = float(name2[1])
        maxim = float(name2[2])
        d2 = round((maxim - minim) / d1, 7)
        #   ниже идет заполнение строк данными о частотах из диапазона частот выше
        data.loc[1] = [minim + i if i == 0 else minim + d2 * i for i in range(d1)]
        #   фильтрация в диапазон 100-400 кГц
        ampl1 = []
        freq1 = []
        for i in range(len(data.loc[1])):
            if maxfre >= data.at[1, i] >= minfre:
                ampl1.append(data.at[0, i])
                freq1.append(data.at[1, i])
        #   Создаем новый отфильтрованный датафрейм
        data = pd.DataFrame([ampl1, freq1])
        d1 = int(len(data.loc[0]))
        minim = float(data.at[1, 0])  # минимум диапазона
        maxim = float(data.at[1, len(data.loc[1]) - 1])  # максимум диапазона


        #   находим максимумы амплитуд и частот и записываем их в датафрейм
        data.loc[2, slice(1, len(data.loc[0]) - 2)] = [
            data.at[0, i] if data.loc[0, i - 1] < data.loc[0, i] > data.loc[0, i + 1]
            else 0 for i in range(1, d1 - 1)]
        data.loc[3, slice(1, len(data.loc[0]) - 2)] = [
            data.at[1, i] if data.loc[0, i - 1] < data.loc[0, i] > data.loc[0, i + 1]
            else 0 for i in range(1, d1 - 1)]


        #   заполняем пустые значения нулями и сохраняем
        data = data.fillna(0)
        #   получаем список со значениями по которым мы будем строить регрессионную модель
        amp = [data.loc[0, i - 2:i + 2] for i in range(2, len(data.loc[0]) - 2) if data.at[3, i] != 0]
        freq = [data.loc[1, i - 2:i + 2] for i in range(2, len(data.loc[0]) - 2) if data.at[3, i] != 0]
        #   Цикл для нахождения всех значений добротности для всего диапазона
        m0 = []  # в этом списке находятся все дельта эф подобранные регрессией используемые при расчете декремента
        Q0 = []  # в этом списке находятся все декременты затухания подобранные регрессией
        freq0 = []  # в этом списке находятся все частоты максимумов на которых расчитаны декременты
        for i in range(len(freq) - 1):
            Z = np.polyfit(freq[i], amp[i], 2)  # не трогай степень полинома!По методу Аблеева идет регрессия параболой!
            if (float(amp[i][2:-2]) / Z[0]) < 0:
                df = math.sqrt(-(float(amp[i][2:-2])) / Z[0])
                Q = df / (float(freq[i][2:-2]))
                m0.append(df)
                Q0.append(Q)
                freq0.append(float(freq[i][2:-2]))
        #        elif (float(amp[i][2:-2])/Z[0]) >=0:
        #            df = math.sqrt(abs(float(amp[i][2:-2])/Z[0]))
        #            Q = df/max(freq[i])
        #            m0.append(df)
        #            Q0.append(Q)
        #            freq0.append(max(freq[i]))

        #  здесь мы находим значения частоты входящие в интервалы в шагом в 2кГц
        a = minim // inter
        k = [0]
        for i in range(len(freq0)):
            #   вот сюда надо добавить переменную чтобы расширить диапазон для регрессии
            if freq0[i] // inter >= a + 1:
                k.append(i)
                a += 1
        #  Создаем список со значениями декремента затухания для каждого интервала
        means.append([np.mean(Q0[k[n - 1]:k[n]]) for n in range(1, len(k))])
        sko.append([np.std(Q0[k[n - 1]:k[n]]) for n in range(1, len(k))])
        freql.append([int(freq0[n]) for n in k])
        del freql[num][0]

        means[num].insert(0, name2[0])
        sko[num].insert(0, name2[0])
        freql[num].insert(0, name2[0])
        #    print(round(float(np.mean(Q0)), 6))
        list13.append(
            [
                [means[num][i] for i in range(len(means[num]))],
                [sko[num][i] for i in range(len(sko[num]))],
                [freql[num][i] for i in range(len(freql[num]))]
            ]
        )

        finalframe = finalframe.append(list13[num], ignore_index=True)
        num += 1
        names.append(name2[0])


# функция которая проводит сортировку по значениям и оставляет только те, что с заданным СКО
def createdata():
    global data0
    global names
    global XYZ
    master = []
    names = list(set(names))  # избавляемся от дубликатов имен
    x = 0
    for n in names:
        where = np.where(finalframe[0] == n)[
            0].tolist()  # в переменной все индексы строк с одинаковым номером СОПа
        # выбираем элементы у которых сумма столбца СКО минимальная для СКО меньше XYZ процента
        o = np.where(np.array([(np.mean(finalframe.loc[where[1::3], i]) / np.mean(finalframe.loc[where[0::3], i]))
                               for i in range(1, len(finalframe.loc[0]) - 1)]) < XYZ)[0].tolist()
        o = [i + 1 for i in o]  # увеличение значения на 1 тк строкой выше слайс был с единицы
        master.append(np.mean(finalframe.loc[where[0::3], o]).tolist())
        # запишем усредненный декремент затухания
        # для каждого интервала, для каждого сопа причем дополнительно запишем имя СОПа
        master.append(finalframe.loc[where[2], o].tolist())
        # найдем и запишем интервалы частот в 2 кГц для которых СКО меньше заданного в XYZ
        master[2 * x].insert(0, n)
        master[2 * x + 1].insert(0, n)
        x += 1
    data0 = pd.DataFrame(master)


# Ниже функция которая ищет пересечения частот, в которых мы находили декремент
# Если частота не входит во все списки, то она и соответствующий ей декремент заменяется на -1
# в дальнейшем функции чекрэйнж и оптимайз можно будет объединить в одну для большего удобства
def checkrange():
    global data0
    k = 0
    for l in range(1, int(len(data0)), 2):
        if data0.loc[l, 1:].count() == max([data0.loc[i, 1:].count() for i in range(1, int(len(data0)), 2)]):
            k = l
    sets = [set(list(data0.loc[l])) for l in range(1, int(len(data0)), 2) if l != k]
    x = list(set(list(data0.loc[k])).intersection(*sets))
    x = sorted(x)
    for i in range(1, len(data0), 2):
        for l in range(1, len(data0.loc[0])):
            if data0.loc[i, l] not in x:
                data0.loc[i, l] = -1
                data0.loc[i - 1, l] = -1


# Ниже функция которая удаляет единицы и форматирует список
def optimize():
    global data0
    lline = []
    for i in range(0, len(data0)):
        y = []
        for j in data0.loc[i]:
            if j in names:
                y.append(j)
            if j not in names:
                if j >= 0:
                    y.append(j)
        lline.append(y)
    data0 = pd.DataFrame(lline)


# построим график декремента от частоты для всех СОПов
def getgraph():
    global data0
    for x in range(0, int((len(data0) - 2) / 2) + 1):
        x0 = list(data0.loc[2 * x, 1:])
        y0 = list(data0.loc[2 * x + 1, 1:])
        z0 = str(data0.at[2 * x, 0])
        plt.scatter(y0, x0, label=z0)
        plt.plot(y0, x0, label=z0)
    plt.legend()
    plt.title('График зависимости декремента затухания от частоты для различных СОПов')
    plt.xlabel('Частота, кГц')
    plt.ylabel('Декремент затухания')
    plt.show()


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


# функция которая строит графики и находит лучший диапазон для измерения (минимальное СКО и макс наклон кривой)
def polypressure():
    global data0
    global Y
    global L
    for x in range(1, len(data0.loc[0]) - 1):
        p = [i for i in data0.loc[0::2, x]]
        Y = [pressure.get(i) for i in data0.loc[0::2, 0]]
        p = np.array(p)  # подготовка к сортировке по декременту затухания
        Y = np.array(Y)  # подготовка к сортировке по давлению гелия
        points = Y.argsort()  # Сортировка по значениям давления гелия
        q = np.polyfit(p[points], Y[points], 1)
        #        print(get_r2_1(p, Y))
        #        print(get_r2_2(p, Y))
        if get_r2_1(p, Y) >= minscore:
            L.append((q[0], q[1], data0.at[1, x]))
        z0 = data0.at[1, x]
        plt.scatter(Y[points], p[points])  # рисуем точки
        plt.plot(Y[points], p[points], label=str((z0, z0 - inter)))  # и график
        plt.plot([q[0] * i + q[1] for i in p[points]], p[points])  # И аппроксимационные прямые
    plt.title('График зависимости декремента затухания от давления гелия для интервала')
    plt.legend()
    plt.xlabel('Давление гелия, атм.')
    plt.ylabel('Декремент затухания')
    plt.show()


# функция которая строит график для лучшего диапазона
def gdef():
    gde = max(np.where(data0 == L[wmax][2])[1].tolist())
    p = [i for i in data0.loc[0::2, gde]]
    Y = [pressure.get(i) for i in data0.loc[0::2, 0]]
    p = np.array(p)  # подготовка к сортировке по давлению
    Y = np.array(Y)  # подготовка к сортировке по чистоте гелия
    points = Y.argsort()  # Сортировка по значениям
    q = np.polyfit(p[points], Y[points], 1)
    #        print(get_r2_1(p, Y))
    #        print(get_r2_2(p, Y))
    plt.scatter(Y[points], p[points])  # рисуем точки
    plt.plot(Y[points], p[points])  # и график
    plt.plot([q[0] * i + q[1] for i in p[points]], p[points])  # И аппроксимационные прямые
    plt.title('Зависимость декремента затухания от давления гелия для СОПов'
              + '{}'.format((data0.loc[1, gde], data0.loc[1, gde] - inter)))
    plt.xlabel('Давление гелия, атм.')
    plt.ylabel('Декремент затухания')
    plt.show()


createfinalframe()
print(finalframe)
createdata()
print(data0)
checkrange()
print(data0)
optimize()
print(data0)
#  запишем отформатированные декременты и их частоты в отдельный файл, с которым потом можно было бы работать
#  finalframe.to_csv(r'finalframe_SOP_100_400_{}.csv'.format(inter), header=False, index=False)
#  data0.to_csv(r'dataSOP_100_400_{}.csv'.format(inter), header=False, index=False)
getgraph()

polypressure()

# находим самые большие по модулю значения коэффициента наклона прямой
fmax = max([abs(L[i][0]) for i in range(len(L))])
# ищем где они расположены
wmax = np.where(fmax == [abs(L[i][0]) for i in range(len(L))])[0].tolist()[0]

gdef()

# выводим на экран финальное сообщение
print('В диапазоне ' + str(L[wmax][2] - inter) + '-' + str(L[wmax][2]) + 'кГц достигается наивысшая чувствительность и '
                                                                         'минимальное СКО. к-ты прямой y = k*x+b : ' + str(
    L[int(wmax)]))
