import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

with open(r'pickle/data/_sop_classic_regr_.pickle', 'rb') as f:
    df = pickle.load(f)
df['delta'] = df.max_int - df.min_int
df = df.loc[df.r2_score > 0]
df.index = [i for i in range(len(df))]

fig = plt.figure(figsize=(10, 6))
ax = sns.distplot(df.r2_score, color='green')
plt.title('Распределение к-та детерминации для \n классического метода в сопах', size=10)
plt.xlabel('К-т детерминации', size=8)
plt.ylabel('Частота', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Классический\Распределение к-та детерминации для классического метода в сопах.png')

fig = plt.figure(figsize=(10, 6))
ax = plt.scatter(df.delta, df.r2_score, color='green', s=0.3, alpha=0.4)
plt.title(
    'Зависимость к-та детерминации аппроксимационной прямой от ширины \n интервала для классического метода в сопах',
    size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('К-т детерминации', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Классический\Зависимость к-та детерминации от ширины интервала для классического метода в сопах.png')

fig = plt.figure(figsize=(10, 6))

ax = plt.scatter(df.delta, df.C_classic, color='green', s=0.3, alpha=0.4)

plt.title('Зависимость наклона аппроксимационной прямой от ширины \n  интервала для классического метода в сопах',
          size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('Наклон аппроксимационной прямой', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Классический\Зависимость наклона аппроксимационной прямой от ширины интервала для классического метода в сопах.png')


#------------------------------------------------------------------------------------------------------------------

with open(r'pickle/data/_sop_heads_regr_.pickle', 'rb') as f:
    df = pickle.load(f)
df['delta'] = df.max_int - df.min_int
df = df.loc[df.r2_score > 0]
df.index = [i for i in range(len(df))]
fig = plt.figure(figsize=(10, 6))
ax = sns.distplot(df.r2_score, color='green')
plt.title('Распределение к-та детерминации для \n метода аппроксимации пиков в сопах', size=10)
plt.xlabel('К-т детерминации', size=8)
plt.ylabel('Частота', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Аппроксимация\Распределение к-та детерминации для метода '
          r'аппроксимации пиков в сопах.png')

fig = plt.figure(figsize=(10, 6))
ax = plt.scatter(df.delta, df.r2_score, color='green', s=0.3, alpha=0.4)
plt.title(
    'Зависимость к-та детерминации аппроксимационной прямой от ширины \n интервала для метода аппроксимации пиков в '
    'сопах',
    size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('К-т детерминации', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Аппроксимация\Зависимость к-та детерминации от ширины интервала '
          r'для метода аппроксимации пиков в сопах.png')

fig = plt.figure(figsize=(10, 6))

ax = plt.scatter(df.delta, df.B_heads, color='green', s=0.3, alpha=0.4)

plt.title('Зависимость наклона аппроксимационной прямой от ширины \n интервала для метода аппроксимации пиков в сопах',
          size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('Наклон аппроксимационной прямой', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Аппроксимация\Зависимость наклона аппроксимационной прямой от '
          r'ширины интервала для метода аппроксимации пиков в сопах.png')

#------------------------------------------------------------------------------------------------------------------


with open(r'pickle/data/_sop_rel_regr_.pickle', 'rb') as f:
    df = pickle.load(f)
df['delta'] = df.max_int - df.min_int
df = df.loc[df.r2_score > 0]
df.index = [i for i in range(len(df))]

fig = plt.figure(figsize=(10, 6))
ax = sns.distplot(df.r2_score, color='green')
plt.title('Распределение к-та детерминации для \n метода рельефности в сопах', size=10)
plt.xlabel('К-т детерминации', size=8)
plt.ylabel('Частота', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Рельефность\Распределение к-та детерминации для метода '
          r'рельефности в сопах.png')

fig = plt.figure(figsize=(10, 6))
ax = plt.scatter(df.delta, df.r2_score, color='green', s=0.3, alpha=0.4)
plt.title(
    'Зависимость к-та детерминации аппроксимационной прямой от ширины \n интервала для метода метода рельефности в '
    'сопах',
    size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('К-т детерминации', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Рельефность\Зависимость к-та детерминации от ширины интервала '
          r'для метода рельефности в сопах.png')

fig = plt.figure(figsize=(10, 6))

ax = plt.scatter(df.delta, df.A_rel, color='green', s=0.3, alpha=0.4)
plt.title('Зависимость наклона аппроксимационной прямой от ширины \n интервала для метода метода рельефности в сопах',
          size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('Наклон аппроксимационной прямой', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\СОПы\Рельефность\Зависимость наклона аппроксимационной прямой от '
          r'ширины интервала для метода метода рельефности в сопах.png')

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


with open(r'pickle/data/_tvel_mox_classic_regr_.pickle', 'rb') as f:
    df = pickle.load(f)
df['delta'] = df.max_int - df.min_int
df = df.loc[df.r2_score > 0]
df.index = [i for i in range(len(df))]

fig = plt.figure(figsize=(10, 6))
ax = sns.distplot(df.r2_score, color='green')
plt.title('Распределение к-та детерминации для \n классического метода в твелах с мокс топливом', size=10)
plt.xlabel('К-т детерминации', size=8)
plt.ylabel('Частота', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Классический\Распределение к-та детерминации для '
          r'классического метода в твелах с мокс топливом.png')

fig = plt.figure(figsize=(10, 6))
ax = plt.scatter(df.delta, df.r2_score, color='green', s=0.3, alpha=0.4)
plt.title(
    'Зависимость к-та детерминации аппроксимационной прямой от ширины \n интервала для классического метода в твелах '
    'с мокс топливом',
    size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('К-т детерминации', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Классический\Зависимость к-та детерминации от ширины '
          r'интервала для классического метода в твелах с мокс топливом.png')

fig = plt.figure(figsize=(10, 6))

ax = plt.scatter(df.delta, df.C_classic, color='green', s=0.3, alpha=0.4)

plt.title('Зависимость наклона аппроксимационной прямой от ширины \n  интервала для классического метода в твелах с '
          'мокс топливом',
          size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('Наклон аппроксимационной прямой', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Классический\Зависимость наклона аппроксимационной прямой '
          r'от ширины интервала для классического метода в твелах с мокс топливом.png')


#------------------------------------------------------------------------------------------------------------------

with open(r'pickle/data/_tvel_mox_heads_regr_.pickle', 'rb') as f:
    df = pickle.load(f)
df['delta'] = df.max_int - df.min_int
df = df.loc[df.r2_score > 0]
df.index = [i for i in range(len(df))]

fig = plt.figure(figsize=(10, 6))
ax = sns.distplot(df.r2_score, color='green')
plt.title('Распределение к-та детерминации для \n метода аппроксимации пиков в твелах с мокс топливом', size=10)
plt.xlabel('К-т детерминации', size=8)
plt.ylabel('Частота', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Аппроксимация\Распределение к-та детерминации для метода '
          r'аппроксимации пиков в твелах с мокс топливом.png')

fig = plt.figure(figsize=(10, 6))
ax = plt.scatter(df.delta, df.r2_score, color='green', s=0.3, alpha=0.4)
plt.title(
    'Зависимость к-та детерминации аппроксимационной прямой от ширины \n интервала для метода аппроксимации пиков в '
    'твелах с мокс топливом',
    size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('К-т детерминации', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Аппроксимация\Зависимость к-та детерминации от ширины '
          r'интервала для метода аппроксимации пиков в твелах с мокс топливом.png')

fig = plt.figure(figsize=(10, 6))

ax = plt.scatter(df.delta, df.B_heads, color='green', s=0.3, alpha=0.4)

plt.title('Зависимость наклона аппроксимационной прямой от ширины \n интервала для метода аппроксимации пиков в '
          'твелах с мокс топливом',
          size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('Наклон аппроксимационной прямой', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Аппроксимация\Зависимость наклона аппроксимационной прямой '
          r'от ширины интервала для метода аппроксимации пиков в твелах с мокс топливом.png')

#------------------------------------------------------------------------------------------------------------------


with open(r'pickle/data/_tvel_mox_rel_regr_.pickle', 'rb') as f:
    df = pickle.load(f)
df['delta'] = df.max_int - df.min_int
df = df.loc[df.r2_score > 0]
df.index = [i for i in range(len(df))]

fig = plt.figure(figsize=(10, 6))
ax = sns.distplot(df.r2_score, color='green')
plt.title('Распределение к-та детерминации для \n метода рельефности в твелах с мокс топливом', size=10)
plt.xlabel('К-т детерминации', size=8)
plt.ylabel('Частота', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Рельефность\Распределение к-та детерминации для метода '
          r'рельефности в твелах с мокс топливом.png')

fig = plt.figure(figsize=(10, 6))
ax = plt.scatter(df.delta, df.r2_score, color='green', s=0.3, alpha=0.4)
plt.title(
    'Зависимость к-та детерминации аппроксимационной прямой от ширины \n интервала для метода метода рельефности в '
    'твелах с мокс топливом',
    size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('К-т детерминации', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Рельефность\Зависимость к-та детерминации от ширины '
          r'интервала для метода рельефности в твелах с мокс топливом.png')

fig = plt.figure(figsize=(10, 6))

ax = plt.scatter(df.delta, df.A_rel, color='green', s=0.3, alpha=0.4)

plt.title('Зависимость наклона аппроксимационной прямой от ширины \n интервала для метода метода рельефности в твелах '
          'с мокс топливом',
          size=10)
plt.xlabel('Ширина интервала, кГц', size=8)
plt.ylabel('Наклон аппроксимационной прямой', size=8)
plt.savefig(
    fname=r'E:\OneDrive\Диплом\Итоговые графики\ТВЕЛ Мокс\Рельефность\Зависимость наклона аппроксимационной прямой от '
          r'ширины интервала для метода метода рельефности в твелах с мокс топливом.png')
