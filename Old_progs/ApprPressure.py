import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def pressure():
    decr = [
        0.001219,
        0.001403,
        0.00162,
        0.001612,
        0.001228,
        0.000472,
        0.000461
    ]

    press = [5, 8, 6, 6, 6, 1, 1]
    name = ['СОП02', 'СОП03', 'СОП05', 'СОП07', 'СОП08', 'ТВЭЛ', 'ТВЭЛ2']
    # Создаем датафрейм и сортируем значения
    data = pd.DataFrame(data=[decr, press], columns=name)
    data1 = data.T
    data1 = data1.sort_values(by=[1])
    # Начинаем строить регрессионную модель
    X = np.array(data1[0]).reshape(-1, 1)
    y = np.array(data1[1])
    poly_reg = PolynomialFeatures(degree=1)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    X_pred = pol_reg.predict(X_poly)
    print(pol_reg.coef_, pol_reg.intercept_, r2_score(data1[0], X_pred))
    listh = [0.000472, 0.000472, 0.000495, 0.000461, 0.000478, 0.000539, 0.001219,
        0.001403, 0.00162, 0.001612, 0.001228, 0.001039]
    New_pred = []
    for a in listh:
        New_pred.append(pol_reg.predict(poly_reg.fit_transform([[a]])))
    for a in New_pred:
        print('Давление гелия ' + str(round(float(a), 2)))
    plt.plot(New_pred, listh)
    plt.plot(data1[1], data1[0])
    plt.show()

def clearity():
    decr = [0.000472, 0.000472, 0.000495, 0.000461, 0.000478, 0.000539]
    clear = [96.2, 96.7, 96.9, 96.6, 96.8, 95.8]
    U = np.polyfit(decr, clear, 2)
    decr = sorted(decr)
    y = [U[2] + U[1] * x + U[0]*x**2 for x in decr]
    plt.plot(decr, y)
    plt.plot(decr, clear)
    plt.show()

# print(pressure())
print(clearity())