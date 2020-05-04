import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv(r'../Остальные данные/СОП3589.csv', header=None)

d1 = int(len(data1.loc[0]))
shag = 5 # int(input(r'введите шаг. кГц/c'))
minim = 100 # int(input('ведите минимум диапазона, кГц'))
maxim = 500 # int(input('ведите максимум диапазона, кГц'))
d2 = round((maxim-minim)/d1,7)
y1 = []
for i in range(0, d1, 1):
    if i == 0:
        y1.append(minim + i)
        i+=1
    else:
        y1.append(minim + d2*i)
        i+=1

fig = plt.figure()
#plt.plot(y1, data1.loc[0], color='black', label='СОП03', lw=0.8)
#plt.plot(y1, data1.loc[1], color='blue', label='СОП05', lw=0.8)
#plt.plot(y1, data1.loc[2], color='red', label='СОП08', lw=0.8)
plt.plot(y1, data1.loc[3], color='yellow', label='СОП09', lw=0.8)
plt.legend()
plt.show()
