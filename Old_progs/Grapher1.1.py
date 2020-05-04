import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv(r'_Частоты СОПов2.csv', header=None, decimal=',', delimiter ='\s+')
fig = plt.figure()

plt.plot(data1.loc[1], data1.loc[17], lw=0.8)
plt.scatter(data1.loc[1], data1.loc[17])
"""
plt.plot(data1.loc[1], data1.loc[10], label='150-200', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[10])

plt.plot(data1.loc[1], data1.loc[11], label='200-250', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[11])

plt.plot(data1.loc[1], data1.loc[12], label='250-300', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[12])

plt.plot(data1.loc[1], data1.loc[13], label='300-350', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[13])

plt.plot(data1.loc[1], data1.loc[14], label='350-400', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[14])

plt.plot(data1.loc[1], data1.loc[15], label='400-450', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[15])

plt.plot(data1.loc[1], data1.loc[16], label='450-500', lw=0.8)
plt.scatter(data1.loc[1], data1.loc[16])
"""
plt.grid(True)
plt.xlabel('Давление гелия, атм.')
plt.ylabel('Фильтрованная ср.добротность для усредненной АЧХ')
plt.title('График зависимости фильтрованной ср.добротности для усредненной АЧХ от давления гелия')
plt.legend()
plt.show()
