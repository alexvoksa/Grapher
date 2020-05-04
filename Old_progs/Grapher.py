
import matplotlib.pyplot as plt
import pandas as pd

#pd.set_option('display.max_columns', None)
data1 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_1.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data2 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_2.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data3 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_3.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data4 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_4.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data5 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_5.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data6 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_6.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data7 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_7.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data8 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_8.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data9 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_9.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data10 = pd.read_csv(r'../АЧХ чистые/СОП-02_0_700_5,0_10.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)

data29 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_11.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data30 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_12.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data31 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_13.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data32 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_14.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data33 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_15.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data34 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_16.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data35 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_17.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data36 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_18.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data37 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_19.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data38 = pd.read_csv(r'../АЧХ чистые/СОП-07_0_700_5,0_20.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)

d1 = int(data1.count(1))

shag = 5 # int(input(r'введите шаг. кГц/c'))
minim = 0 # int(input('ведите минимум диапазона, кГц'))
maxim = 700 # int(input('ведите максимум диапазона, кГц'))
d2 = round((maxim-minim)/d1, 7)

y1 = []
for i in range(0, d1, 1):
    if i == 0:
        y1.append(minim + i)
        i+=1
    else:
        y1.append(minim + d2*i)
        i+=1

# ниже создание массивов для преобразований и нахождения среднего

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x10 = []

x29 = []
x30 = []
x31 = []
x32 = []
x33 = []
x34 = []
x35 = []
x36 = []
x37 = []
x38 = []

for j in range(0, d1,1):
    x1.append(data1.at[0,j])
    x2.append(data2.at[0,j])
    x3.append(data3.at[0,j])
    x4.append(data4.at[0,j])
    x5.append(data5.at[0,j])
    x6.append(data6.at[0,j])
    x7.append(data7.at[0,j])
    x8.append(data8.at[0,j])
    x9.append(data9.at[0,j])
    x10.append(data10.at[0,j])

    x29.append(data29.at[0, j])
    x30.append(data30.at[0, j])
    x31.append(data31.at[0, j])
    x32.append(data32.at[0, j])
    x33.append(data33.at[0, j])
    x34.append(data34.at[0, j])
    x35.append(data35.at[0, j])
    x36.append(data36.at[0, j])
    x37.append(data37.at[0, j])
    x38.append(data38.at[0, j])

x = pd.DataFrame([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], index=None, columns=None) # массив для соп 02
xx = pd.DataFrame([x29, x30, x31 ,x32 ,x33 ,x34 ,x35 ,x36 ,x37 ,x38], index=None, columns=None) # массив для соп 07


x0 = [] # функция которая будет усреднять для СОП-03
x00 = [] # функция которая будет усреднять для СОП-09
# xmid = [] # функция которая будет вычитать из соп 09 соп 03



for m in range(0, d1, 1):
    x0.append(round(x[m].mean(),4))
    x00.append(round(xx[m].mean(),4))
#    zz = round(x00[m] - x0[m],4)
#    if zz > 0:
#        zz = zz
#    else: zz = 0
#    xmid.append(zz)

#print(xmid)
#file = open('вычитание03из09.csv','w')
#file.write(str(xmid))
#file.close()

file1 = open('../Остальные данные/соп09.csv', 'w')
file1.write(str(x00))
file1.close()

file3 = open('соп03.csv','w')
file3.write(str(x0))
file3.close()


"""p = [] #цикл для нахождения точек максимумов по координатам
for w in range(0, d1, 1):
    if x0[w] < x0[(w+1)] > x0[(w+2)]:
        p.append(x0[(w+1)])
    else:
        w+=1


print(p)

"""
fig = plt.figure()
print(fig.axes)
print(type(fig))
plt.plot(y1, xmid)
plt.show()
print(fig.axes)
