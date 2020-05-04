import pandas as pd
import numpy as np

complist = []
pressure = {
    'СОП-02': 5,
    'СОП-03': 8,
    'СОП-05': 6,
    'СОП-07': 6,
    'СОП-08': 6
}  # в этом словаре находятся названия файлов и их давления или чистота гелия


def getdata0():
    global data0
    data = pd.read_csv(r'rel_finalframe_SOP_100_500_2.csv', header=None)
    data = data.dropna(axis=1)
    xframe = []
    yframe = []
    for i in pressure:
        data0 = pd.DataFrame([data.loc[j].tolist() for j in range(len(data)) if data.loc[j, 0] == i])
        xlen = np.mean(data0.loc[0::2, 1:], axis=0).tolist()
        xlen.insert(0, i)
        xframe.append(xlen)
        ylen = np.mean(data0.loc[1::2, 1:], axis=0).tolist()
        ylen.insert(0, i)
        yframe.append(ylen)
    data0 = pd.DataFrame(xframe)
    data0 = data0.append(yframe)
    data0 = data0.reset_index(drop=True)


def getlist():
    global pressure
    global data0
    global complist
    for i in pressure:
        ext, zz = i.split('-')
        zz = int(zz)
        compare = []
        j = np.where(data0[0] == i)[0].tolist()
#        if i == 'СОП-02' or i == 'СОП-07':
#            data0.loc[j[1], 1:] = data0.loc[np.where(data0[0] == 'СОП-05')[0].tolist()[1], 1:]

        for k in range(2, len(data0.loc[j[0], 1:])):
            m = 1
            for y in range(2, len(data0.loc[j[0], 1:]), k):
                diff = np.diff(data0.loc[j[0], m: y].to_list()) / np.diff(data0.loc[j[1], m: y].to_list())

                rel = round(np.trapz(abs(diff), x=data0.loc[j[1], m+1:y].to_list()) /
                            np.trapz(data0.loc[j[0], m:y].to_list(), x=data0.loc[j[1], m:y].to_list()), 5)

                press = int(pressure.get(i))
                intermax = round(float(data0.loc[j[1], y]), 4)
                intermin = round(float(data0.loc[j[1], m]), 4)
                spec = (zz, rel, press, intermax, intermin)
                print(spec)
                compare.append(spec)
                m = y

        megadata = pd.DataFrame(compare)
        megadata.to_csv('megadata{}.csv'.format(i), header=False, index=False)

getdata0()
data0.dropna(axis=1)
data0.to_csv('data0_rel.csv', header=False, index=False)
# getlist()