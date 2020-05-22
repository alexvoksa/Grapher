import pandas as pd
import numpy as np
import re
import os.path
from tqdm import tqdm
import os
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

with open('constant_frame.pickle', 'rb') as file:
    constant_frame = pickle.load(file)
corr_df = pd.DataFrame(data=None, columns=['feature', 'rel', 'heads', 'classic'])
for i in range(len(constant_frame)):
    corr_frame = pd.DataFrame([np.array(constant_frame.iloc[i, 1::9]), np.array(constant_frame.iloc[i, 4::9]),
                               np.array(constant_frame.iloc[i, 5::9]), np.array(constant_frame.iloc[i, 7::9])]).T
    corr_frame.columns = ['feature', 'rel', 'heads', 'classic']
    c_df = corr_frame.corr()
    corr_df = pd.concat([corr_df, c_df.iloc[0:1]], axis=0, ignore_index=True)

from scipy import stats

stats.pearsonr()
