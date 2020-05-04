import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
data2 = pd.read_csv(r'../Добротность/СОП02_добротностьF.csv', header=None)
data3 = pd.read_csv(r'../Добротность/СОП03_добротностьF.csv', header=None)
data5 = pd.read_csv(r'../Добротность/СОП05_добротностьF.csv', header=None)
data7 = pd.read_csv(r'../Добротность/СОП07_добротностьF.csv', header=None)
data8 = pd.read_csv(r'../Добротность/СОП08_добротностьF.csv', header=None)
data9 = pd.read_csv(r'../Добротность/СОП09_добротностьF.csv', header=None)

# хранить цикл ниже как зеницу ока. Расчет средней добротности для интервала
dip = pd.DataFrame()
st = 0
q = 0
j = 0
sr = []
for k in range(100, 500, 50):
    for i in range(0, len(data2.loc[0]), 1):
        if k <= data2.at[1, i] <= k+50:
            sr.append(i)
        else:
            i += 1
    k = sr[0]
    l = sr[-1]
    m = round(float(np.mean(data2.loc[0, k:l])), 4)
    dip.at[st, q] = m
    q += 1
st += 1
q = 0
j = 0
sr = []
for k in range(100, 500, 50):
    for i in range(0, len(data3.loc[0]), 1):
        if k <= data3.at[1, i] <= k+50:
            sr.append(i)
        else:
            i += 1
    k = sr[0]
    l = sr[-1]
    m = round(float(np.mean(data3.loc[0, k:l])), 4)
    dip.at[st, q] = m
    q += 1
st += 1
q = 0
j = 0
sr = []
for k in range(100, 500, 50):
    for i in range(0, len(data5.loc[0]), 1):
        if k <= data5.at[1, i] <= k+50:
            sr.append(i)
        else:
            i += 1
    k = sr[0]
    l = sr[-1]
    m = round(float(np.mean(data5.loc[0, k:l])), 4)
    dip.at[st, q] = m
    q += 1
st += 1
q = 0
j = 0
sr = []
for k in range(100, 500, 50):
    for i in range(0, len(data7.loc[0]), 1):
        if k <= data7.at[1, i] <= k+50:
            sr.append(i)
        else:
            i += 1
    k = sr[0]
    l = sr[-1]
    m = round(float(np.mean(data7.loc[0, k:l])), 4)
    dip.at[st, q] = m
    q += 1
st += 1
q = 0
j = 0
sr = []
for k in range(100, 500, 50):
    for i in range(0, len(data8.loc[0]), 1):
        if k <= data8.at[1, i] <= k+50:
            sr.append(i)
        else:
            i += 1
    k = sr[0]
    l = sr[-1]
    m = round(float(np.mean(data8.loc[0, k:l])), 4)
    dip.at[st, q] = m
    q += 1
st += 1
q = 0
j = 0
sr = []
for k in range(100, 500, 50):
    for i in range(0, len(data9.loc[0]), 1):
        if k <= data9.at[1, i] <= k+50:
            sr.append(i)
        else:
            i += 1
    k = sr[0]
    l = sr[-1]
    m = round(float(np.mean(data9.loc[0, k:l])), 4)
    dip.at[st, q] = m
    q += 1
st += 1



"""
for k in range(150, 500, 50):
    for i in range(1, len(data4.loc[0]), 1):
        if (k-50 <= data4.at[1, i] <= k) and (data4.at[1, i+1] > k or i+1> len(data4.loc[0])):
            m = round(float(np.mean(data4.loc[0, j:i])), 4)
            j = i
            dip.at[st, q] = m
            q += 1
        elif i+1 > len(data4.loc[0]):
                q+=1
                m = round(float(np.mean(data4.loc[0, j:len(data4.loc[0])])), 4)
                dip.at[st, q] = m
                break
        else:
            dip.at[st,q] = 0
st += 1
q = 0
j = 0


y1 = np.array([8, 6, 5, 0]).reshape(-1, 1)

# у соп 05 другие параметры компенсационного объема!!!

x1 = np.array([dip.at[0, 0], dip.at[2, 0], dip.at[4, 0], dip.at[3, 0]]) # 100-150
x2 = np.array([dip.at[0, 1], dip.at[2, 1], dip.at[4, 1], dip.at[3, 1]]) # 150-200
x3 = np.array([dip.at[0, 2], dip.at[2, 2], dip.at[4, 2], dip.at[3, 2]]) # 200-250
x4 = np.array([dip.at[0, 3], dip.at[2, 3], dip.at[4, 3], dip.at[3, 3]]) # 250-300
x5 = np.array([dip.at[0, 4], dip.at[2, 4], dip.at[4, 4], dip.at[3, 4]]) # 300-350
x6 = np.array([dip.at[0, 5], dip.at[2, 5], dip.at[4, 5], dip.at[3, 5]]) # 350-400
x7 = np.array([dip.at[0, 6], dip.at[2, 6], dip.at[4, 6], dip.at[3, 6]]) # 400-450
x8 = np.array([dip.at[0, 7], dip.at[2, 7], dip.at[4, 7], dip.at[3, 7]]) # 450-500

# решение задачи линейной регрессии
skm = lm.LinearRegression()
skm.fit(y1,x1)
pred1 = skm.predict(y1)

print('100-150 ' + str(skm.intercept_) + str(skm.coef_))

skm = lm.LinearRegression()
skm.fit(y1,x2)
pred2 = skm.predict(y1)


skm = lm.LinearRegression()
skm.fit(y1,x3)
pred3 = skm.predict(y1)

skm = lm.LinearRegression()
skm.fit(y1,x4)
pred4 = skm.predict(y1)

skm = lm.LinearRegression()
skm.fit(y1,x5)
pred5 = skm.predict(y1)

skm = lm.LinearRegression()
skm.fit(y1,x6)
pred6 = skm.predict(y1)

skm = lm.LinearRegression()
skm.fit(y1,x7)
pred7 = skm.predict(y1)

skm = lm.LinearRegression()
skm.fit(y1,x8)
pred8 = skm.predict(y1)
print('450-500' + str(skm.intercept_) + str(skm.coef_))



fig, ax = plt.subplots()
plt.plot(y1, x1, label='100-150 кГц', lw=1)
plt.scatter(y1, x1)

# plt.plot(dip[1], dip[1], color='black', label='Добротность СОП05', lw=0.8)

ax.plot(y1, x2, label='150-200 кГц', lw=1)
plt.scatter(y1,  x2)

ax.plot(y1, x3, label='200-250 кГц', lw=1)
plt.scatter(y1, x3)

ax.plot(y1, x4, label='250-300 кГц', lw=1)
plt.scatter(y1,  x4)

ax.plot(y1, x5, label='300-350 кГц', lw=1)
plt.scatter(y1,  x5)

ax.plot(y1, x6, label='350-400 кГц', lw=1)
plt.scatter(y1,  x6)

ax.plot(y1, x7, label='400-450 кГц', lw=1)
plt.scatter(y1,  x7)

ax.plot(y1, x8, label='450-500 кГц', lw=1)
plt.scatter(y1, x8)

#ax.plot(y1, pred1, color='yellow')
#ax.plot(y1, pred2, color='yellow')
#ax.plot(y1, pred3, color='yellow')
#ax.plot(y1, pred4, color='yellow')
#ax.plot(y1, pred5, color='yellow')
#ax.plot(y1, pred6, color='yellow')
#ax.plot(y1, pred7, color='yellow')
#ax.plot(y1, pred8, color='yellow')



ax.grid()
ax.set_xlabel('Давление гелия, атм')
ax.set_ylabel('Средняя добротность для интервала')

plt.legend(bbox_to_anchor=(1, 1))
plt.show()

"""
print(dip)

for i in range(0, 7):
    print(
    'Для интервала ' + str(i)+' среднее значение добротности ' + str(round(float(np.mean(dip[i])), 3)) + ' с СКО ' + str(round((100*np.std(dip[i])/np.mean(dip[i])), 2) )+ '%')
