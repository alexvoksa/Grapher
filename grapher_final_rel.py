import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('final_relefnost_SOP.csv', header=None) # для давления гелия

data = data.dropna(axis=0)
data.index = [i for i in range(len(data))]
data = data.drop(np.where(data[0] == 0)[0].tolist())
data.index = [i for i in range(len(data))]

# i1 = int(32) # для чистоты гелия
# i2 = int(33) # для чистоты гелия
i1 = int(37)   # для давления гелия
i2 = int(38)   # для давления гелия


a = np.where(data[39] <= 0.05)[0].tolist()  # для давления гелия
b = np.where(data[37] >= 0.9)[0].tolist()   # для давления гелия
c = list(set(a) & set(b))[0]

xm = c


y = data.loc[xm, 1:(i1-3):7].to_list()  # для давления гелия
x = data.loc[xm, 2:(i1-3):7].to_list()  # для давления гелия

deter = data.loc[xm, i1]
par1 = data.loc[xm, i1-2]
sko1 = np.mean(data.loc[xm, 5:(i1-1):7])
sko2 = np.std(data.loc[xm, 5:(i1-1):7])
sko = (data.loc[xm, 1:(i1-2):7].tolist(), data.loc[xm, 5:(i1-1):7].tolist(), data.loc[xm, 0:(i1-3):7].tolist())


k1 = float(data.loc[xm, i1-2].split(',')[0].replace('[', '').replace(' ', ''))  # для давления гелия
k2 = float(data.loc[xm, i1-2].split(',')[1].replace(']', '').replace(' ', ''))

print('Зависимость рельефности от давления гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
            str(data.at[xm, 4]) + 'кГц')
print('Из-за того, что при снятии АЧХ были использованы различные диапазоны измерений \n '
      'погрешность определения интервала составила +/- 40Гц')
print('К-т детерминации: ', deter)
print('Параметны прямой: ', par1)
print('Среднее СКО рельефности: ', sko1)  # для давления
print('CКО определения среднего СКО: ', sko2)  # для давления
print('Значение рельефности и ее CКО для каждого СОПа: ',
      [(sko[0][i], sko[1][i], sko[2][i]) for i in range(len(sko[0]))])  # для давления


plt.plot(x, [x*(k1) + k2 for x in x], color='yellow')
plt.scatter(x, y)
#   plt.xlabel('Чистота гелия, проценты') # для чистоты гелия
plt.xlabel('Давление гелия, атм.') # для давления гелия
plt.ylabel('Рельефность,')
# plt.title('График зависимости рельефности от чистоты гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
#          str(data.at[xm, 4]) + 'кГц')
plt.title('График зависимости рельефности от давления гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
          str(data.at[xm, 4]) + 'кГц. ТВЭЛы БН')
plt.show()



deter = data.loc[xm, 30]
par1 = data.loc[xm, 28]

k1 = float(data.loc[xm, 28].split(',')[0].replace('[', '').replace(' ', ''))  # для давления гелия
k2 = float(data.loc[xm, 28].split(',')[1].replace(']', '').replace(' ', ''))
y = data.loc[xm, 1:(28):7].to_list()  # для давления гелия
x = data.loc[xm, 2:(28):7].to_list()  # для давления гелия
plt.plot(list(np.linspace(97, 99, 100)), list(np.linspace(97, 99, 100)*(k1) + k2), color='yellow')
plt.scatter(x, y)
plt.xlabel('Чистота гелия, проценты') # для чистоты гелия
#     plt.xlabel('Давление гелия, атм.') # для давления гелия
plt.ylabel('Рельефность,')
# plt.title('График зависимости рельефности от чистоты гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
#          str(data.at[xm, 4]) + 'кГц')
plt.title('График зависимости рельефности от чистоты гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
          str(data.at[xm, 4]) + 'кГц. ТВЭЛы БН\n'
                                'среднее относительное СКО менее 1.3 % \n')
plt.show()
print('Зависимость рельефности от чистоты гелия в диапазоне ' + str(data.at[xm, 3]) + ' - ' +
            str(data.at[xm, 4]) + 'кГц')
print('К-т детерминации: ', deter)
print('Параметры прямой: ', par1)
print('Среднее относительное СКО, %', round(data.loc[xm, 32], 3)*100)