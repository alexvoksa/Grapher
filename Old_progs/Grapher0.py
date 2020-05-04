import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_17.txt', delimiter='\s+', header=None, nrows=1, decimal=',')
y = data.loc[0].to_list()
data.loc[1] = [100+(400/len(data.loc[0]))*i for i in range(len(data.loc[0]))]
x = [i for i in data.loc[1]]
rel = np.trapz(abs(np.diff(y)/np.diff(x)), x=x[1::1])/np.trapz(y, x=x)
print(rel)
plt.plot(x[1:], abs(np.diff(y)/np.diff(x)), label='График модуля производной функции')
plt.plot(x, y, label='АЧХ СОПа')
plt.legend()
plt.title("Совмещенный график производной и АЧХ СОПа")
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, попугаи')

pressure = {
    'СОП-02': 5,
    'СОП-03': 8,
    'СОП-05': 6,
    'СОП-07': 6,
    'СОП-08': 6
}   # в этом словаре находятся названия файлов и их давления или чистота гелия

Y = []
L = []

XYZ = float(0.4)  # в этой переменной записано наибольшее значение СКО интервала для декремента затухания
inter = int(2)  # в этой переменной записан шаг с которым мы проходим по общему диапазону
minscore = 0.6  # в этой переменной записано наименьшее допустимое значение к-та R^2

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
        plt.plot(Y[points], p[points], label=str((z0, z0-inter)))  # и график
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
    plt.title('Зависимость декремента затухания от давления гелия для проколов СОПов'
              + '{}'.format((data0.loc[1, gde], data0.loc[1, gde] - inter)))
    plt.xlabel('Давление гелия, атм.')
    plt.ylabel('Декремент затухания')
    plt.show()


getgraph()
polypressure()
# находим самые большие по модулю значения коэффициента наклона прямой
fmax = max([abs(L[i][0]) for i in range(len(L))])
# ищем где они расположены
wmax = np.where(fmax == [abs(L[i][0]) for i in range(len(L))])[0].tolist()[0]
# выводим на экран финальное сообщение
print(L)
print(L[wmax])
print(wmax)
print(fmax)


gdef()
print('В диапазоне ' + str(L[wmax][2] - 2) + '-' + str(L[wmax][2]) + 'кГц достигается наивысшая чувствительность'
                                                                     '. к-ты прямой y = k*x+b : ' + str(
    L[int(wmax)]) + 'СКО не превышает ' + str(0.5*100) + '%')
"""


purity = {
    '00203C4V': 96.2,
    '00282C4V': 96.7,
    '00359C4V1': 96.9,
    '00378C4V': 96.6,
    '00381C4V1': 96.8,
    '00397C4V1': 95.8
}  # в этом словаре находятся названия файлов и их давления или чистота гелия
Y = []
L = []
XYZ = float(0.9)  # в этой переменной записано наибольшее значение СКО интервала для декремента затухания
inter = int(40)
minscore = 0.5
lis = [
    r'Реальные ТВЭЛы/00203C4V_10_400_1,0_1.txt',
    r'Реальные ТВЭЛы/00282C4V_10_400_1,0_11.txt',
    r'Реальные ТВЭЛы/00359C4V1_10_400_1,0_7.txt',
    r'Реальные ТВЭЛы/00378C4V_10_400_1,0_12.txt',
    r'Реальные ТВЭЛы/00381C4V1_10_400_1,0_10.txt',
    r'Реальные ТВЭЛы/00397C4V1_10_400_1,0_6.txt'
]  # это список с путем к исследоваемым файлам ВНИМАНИЕ путь к файлу должен быть указан строго как здесь
names = []
finalframe = pd.read_csv(r'finalframe_TVEL_100_400_40.csv', header=None)
data0 = pd.read_csv(r'dataTVEL_0_400_40.csv', header=None)

minfre = int(0)  # задание переменной которая отвечает за нижн границу диапазона к которому будут приводиться данные
maxfre = int(400)  # задание переменной которая отвечает за верх границу диапазона к которому будут приводиться данные


# функция которая проводит сортировку по значениям и осставляет только те, что с заданным СКО
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
        #        o = np.where((finalframe.loc[where[1::3], 1:].sum(axis=0) / finalframe.loc[where[0::3], 1:].sum(axis=0))
        #                     < XYZ)[0].tolist()
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
    plt.title('График зависимости декремента затухания от частоты для различных ТВЭЛов')
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
def polypurity():
    global data0
    global Y
    global L
    global minscore
    slist = []
    for x in range(1, len(data0.loc[0]) - 1):
        p = [i for i in data0.loc[0::2, x]]
        Y = [purity.get(i) for i in data0.loc[0::2, 0]]
        p = np.array(p)  # подготовка к сортировке по декременту затухания
        Y = np.array(Y)  # подготовка к сортировке по чистоте гелия
        points = Y.argsort()  # Сортировка по значениям
        q = np.polyfit(p[points], Y[points], 1)
        #if q[0] < 0:
        #    slist.append(data0.loc[1, x])
        #        print(get_r2_1(p, Y))
        #        print(get_r2_2(p, Y))
        if get_r2_1(p, Y) >= minscore:
            L.append((q[0], q[1], data0.at[1, x]))
        z0 = str(data0.at[1, x])
        plt.scatter(Y[points], p[points], label=z0 )  # рисуем точки
        plt.plot(Y[points], p[points])  # и график
        plt.plot([q[0] * i + q[1] for i in p[points]], p[points])  # И аппроксимационные прямые
        plt.legend()
    plt.title('График зависимости декремента затухания от чистоты гелия для проколов ТВЭЛов')
    plt.xlabel('Чистота гелия, проценты.')
    plt.ylabel('Декремент затухания')
    plt.show()



# функция которая строит график для лучшего диапазона
def gdef():
    gde = max(np.where(data0 == L[wmax][2])[1].tolist())
    p = [i for i in data0.loc[0::2, gde]]
    Y = [purity.get(i) for i in data0.loc[0::2, 0]]
    p = np.array(p)  # подготовка к сортировке по давлению
    Y = np.array(Y)  # подготовка к сортировке по чистоте гелия
    points = Y.argsort()  # Сортировка по значениям
    q = np.polyfit(p[points], Y[points], 1)
    #        print(get_r2_1(p, Y))
    #        print(get_r2_2(p, Y))
    plt.scatter(Y[points], p[points])  # рисуем точки
    plt.plot(Y[points], p[points])  # и график
    plt.plot([q[0] * i + q[1] for i in p[points]], p[points])  # И аппроксимационные прямые
    plt.title('Зависимость декремента затухания от чистоты гелия для проколов ТВЭЛов'
              + '{}'.format((data0.loc[1, gde], data0.loc[1, gde] - 2)))
    plt.xlabel('Чистота гелия, проценты.')
    plt.ylabel('Декремент затухания')
    plt.show()


#data0 = pd.read_csv(r'dataTVEL_0_400.csv', header=None)
createdata()
print(data0)
checkrange()
print(data0)
optimize()
print(data0)

getgraph()
polypurity()

# находим самые большие по модулю значения коэффициента наклона прямой
fmax = max([abs(L[i][0]) for i in range(len(L))])
# ищем где они расположены
wmax = np.where(fmax == [abs(L[i][0]) for i in range(len(L))])[0].tolist()[0]
# выводим на экран финальное сообщение
print(L)
print(L[wmax])
print(wmax)
print(fmax)


def gdef():
    gde = max(np.where(data0 == L[wmax][2])[1].tolist())
    p = [i for i in data0.loc[0::2, gde]]
    Y = [purity.get(i) for i in data0.loc[0::2, 0]]
    p = np.array(p)  # подготовка к сортировке по давлению
    Y = np.array(Y)  # подготовка к сортировке по чистоте гелия
    points = Y.argsort()  # Сортировка по значениям
    q = np.polyfit(p[points], Y[points], 1)
    #        print(get_r2_1(p, Y))
    #        print(get_r2_2(p, Y))
    plt.scatter(Y[points], p[points])  # рисуем точки
    plt.plot(Y[points], p[points])  # и график
    plt.plot([q[0] * i + q[1] for i in p[points]], p[points])  # И аппроксимационные прямые
    plt.title('Зависимость декремента затухания от чистоты гелия для проколов ТВЭЛов'
              + '{}'.format((data0.loc[1, gde], data0.loc[1, gde] - 2)))
    plt.xlabel('Чистота гелия, проценты.')
    plt.ylabel('Декремент затухания')
    plt.show()


gdef()
print('В диапазоне ' + str(L[wmax][2] - 2) + '-' + str(L[wmax][2]) + 'кГц достигается наивысшая чувствительность'
                                                                     '. к-ты прямой y = k*x+b : ' + str(
    L[int(wmax)]) + 'СКО не превышает ' + str(0.5 * 100) + '%')
"""