import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

finalframe = pd.read_csv(r'Old_progs/rel_finalframe_SOP_100_500_2.csv', header=None)
print(finalframe)

diff = np.diff(data0.loc[j[0], 1:].to_list()) / np.diff(data0.loc[j[1], 1:].to_list())

rel = round(np.trapz(abs(diff), x=data0.loc[j[1], 2:].to_list()) /
            np.trapz(data0.loc[j[0], m:y].to_list(), x=data0.loc[j[1], m:].to_list()), 5)