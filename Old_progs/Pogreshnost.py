import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
# СОП03
data1 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_1.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data2 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_2.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data3 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_3.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data4 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_4.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data5 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_5.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data6 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_6.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data7 = pd.read_csv(r'../АЧХ чистые/СОП-03_100_500_5,0_7.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
# СОП05
data8 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_8.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data9 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_9.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data10 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_10.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data11 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_11.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data12 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_12.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data13 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_13.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data14 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_14.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data15 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_15.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data16 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_16.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data17 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_17.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data18 = pd.read_csv(r'../АЧХ чистые/СОП-05_100_500_5,0_18.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
# СОП08
data19 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_19.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data20 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_20.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data21 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_21.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data22 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_22.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data23 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_23.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data24 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_24.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data25 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_25.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data26 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_26.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data27 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_27.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data28 = pd.read_csv(r'../АЧХ чистые/СОП-08_100_500_5,0_28.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
# СОП09
data29 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_29.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data30 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_30.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data31 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_31.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data32 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_32.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data33 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_33.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data34 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_34.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data35 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_35.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data36 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_36.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data37 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_37.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)
data38 = pd.read_csv(r'../АЧХ чистые/СОП-09_100_500_5,0_38.txt', decimal=',', header=None, delimiter=r"\s+", nrows=1)


d1 = int(data1.count(1))

shag = 5  # int(input(r'введите шаг. кГц/c'))
minim = 100  # int(input('ведите минимум диапазона, кГц'))
maxim = 500  # int(input('ведите максимум диапазона, кГц'))
d2 = round((maxim-minim)/d1, 7)

y1 = []
for i in range(0, d1, 1):
    if i == 0:
        y1.append(minim + i)
        i += 1
    else:
        y1.append(minim + d2*i)
        i += 1

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
x11 = []
x12 = []
x13 = []
x14 = []
x15 = []
x16 = []
x17 = []
x18 = []
x19 = []
x20 = []
x21 = []
x22 = []
x23 = []
x24 = []
x25 = []
x26 = []
x27 = []
x28 = []
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

for j in range(0, d1, 1):
    x1.append(data1.at[0, j])
    x2.append(data2.at[0, j])
    x3.append(data3.at[0, j])
    x4.append(data4.at[0, j])
    x5.append(data5.at[0, j])
    x6.append(data6.at[0, j])
    x7.append(data7.at[0, j])
    x8.append(data8.at[0, j])
    x9.append(data9.at[0, j])
    x10.append(data10.at[0, j])
    x11.append(data11.at[0, j])
    x12.append(data12.at[0, j])
    x13.append(data13.at[0, j])
    x14.append(data14.at[0, j])

    x15.append(data15.at[0, j])
    x16.append(data16.at[0, j])
    x17.append(data17.at[0, j])
    x18.append(data18.at[0, j])
    x19.append(data19.at[0, j])
    x20.append(data20.at[0, j])
    x21.append(data21.at[0, j])
    x22.append(data22.at[0, j])
    x23.append(data23.at[0, j])
    x24.append(data24.at[0, j])
    x25.append(data25.at[0, j])
    x26.append(data26.at[0, j])
    x27.append(data27.at[0, j])
    x28.append(data28.at[0, j])

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

x03 = pd.DataFrame([x1, x2, x3, x4, x5, x6, x7])  # массив для соп 03
x05 = pd.DataFrame([x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18])  # массив для соп 05
x08 = pd.DataFrame([x19, x20, x21, x22, x23, x24, x25, x26, x27, x28])  # массив для соп 08
x09 = pd.DataFrame([x29, x30, x31, x32, x33, x34, x35, x36, x37, x38])  # массив для соп 09



# x0 = [] # функция которая будет усреднять для СОП-03
# x00 = [] # функция которая будет усреднять для СОП-03
# x000 = [] # функция которая будет усреднять для СОП-03
# x0000 = [] # функция которая будет усреднять для СОП-09

# for m in range(0, d1, 1):
#    x0.append(round(x03[m].mean(), 4))
#    x00.append(round(x05[m].mean(), 4))
#    x000.append(round(x08[m].mean(), 4))
#    x0000.append(round(x09[m].mean(), 4))

x03.to_csv('СОП03 массив.csv', index=False)
x05.to_csv('СОП05 массив.csv', index=False)
x08.to_csv('СОП08 массив.csv', index=False)
x09.to_csv('СОП09 массив.csv', index=False)


"""
sko03 = []
sko05 = []
sko08 = []
sko09 = []

mea03 = []
mea05 = []
mea08 = []
mea09 = []

otnos03 = []
otnos05 = []
otnos08 = []
otnos09 = []

for i in range(0, d1, 1):
    sko03.append(np.std(x03, axis=0)[i])
    mea03.append(np.mean(x03, axis=0)[i])
for i in range(0, d1, 1):
    sko05.append(np.std(x05, axis=0)[i])
    mea05.append(np.mean(x05, axis=0)[i])
for i in range(0, d1, 1):
    sko08.append(np.std(x08, axis=0)[i])
    mea08.append(np.mean(x08, axis=0)[i])
for i in range(0, d1, 1):
    sko09.append(np.std(x09, axis=0)[i])
    mea09.append(np.mean(x09, axis=0)[i])

for i in range(0, d1, 1):
    otnos03.append(sko03[i] / mea03[i])
    otnos05.append(sko05[i] / mea05[i])
    otnos08.append(sko08[i] / mea08[i])
    otnos09.append(sko09[i] / mea09[i])

p03 = str(round(float(np.mean(otnos03)), 2)*100)
p05 = str(round(float(np.mean(otnos05)), 2)*100)
p08 = str(round(float(np.mean(otnos08)), 2)*100)
p09 = str(round(float(np.mean(otnos09)), 2)*100)

print('Относительная погрешность измерения СОП03 составила ' + p03 + '%')
print('Относительная погрешность измерения СОП05 составила ' + p05 + '%')
print('Относительная погрешность измерения СОП08 составила ' + p08 + '%')
print('Относительная погрешность измерения СОП09 составила ' + p09 + '%')
"""