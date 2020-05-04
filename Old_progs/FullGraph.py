import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Данные по сопам ВВЭР-440 01-04-20.csv', decimal=',', delimiter=';', header=None)

plt.plot(data.loc[1], data.loc[4], label='Рельефность')
#plt.bar(data.loc[1], data.loc[5], label='Усредненная ср.добротность')
plt.plot(data.loc[1], data.loc[6], label='Декремент затухания +/- 10')
plt.plot(data.loc[1], data.loc[7], label='Декремент затухания +/- 5')
#plt.bar(data.loc[1], data.loc[8], label='ср.добротность для усредненной АЧХ без фильтра')
#plt.bar(data.loc[1], data.loc[9], label='ср.добротность для усредненной АЧХ с фильтром')

plt.grid()
plt.title("Зависимость значений методов от давления гелия")
plt.legend()
plt.xlabel('Давление, атм')
plt.ylabel('Значение')
plt.show()