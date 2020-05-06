import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from tqdm import tqdm
from multiprocessing import Pool
import math
import timeit


def classic_decr_with_np(amps_in_range, freq_in_range):
    # structure ['coord', 'amp_max', 'freq_max', 'freq_min_1', 'freq_min_2', 'found_freq_left', 'found_freq_right']
    coords_amps_max = argrelextrema(amps_in_range, np.greater)[0]
    max_amps_in_range = np.array(amps_in_range[coords_amps_max])
    max_freq_in_range = np.array(freq_in_range[coords_amps_max])
    lower_range_amps = max_amps_in_range * 1 / math.sqrt(2) * 0.95
    upper_range_amps = max_amps_in_range * 1 / math.sqrt(2) * 1.05
    list_of_decr = []
    for i in range(len(coords_amps_max)):
        values_min = []
        values_max = []
        for j in range(55):
            if int(coords_amps_max[i]) - j >= 0:
                if lower_range_amps[i] <= amps_in_range[int(coords_amps_max[i]) - j] <= upper_range_amps[i]:
                    values_min.append(freq_in_range[int(coords_amps_max[i]) - j])
        for j in range(55):
            if int(coords_amps_max[i]) + j <= len(coords_amps_max):
                if lower_range_amps[i] <= amps_in_range[int(coords_amps_max[i]) + j] <= upper_range_amps[i]:
                    values_max.append(freq_in_range[int(coords_amps_max[i]) + j])
        if len(values_min) > 0:
            values_min = values_min[0]
        else:
            values_min = 0
        if len(values_max) > 0:
            values_max = values_max[0]
        else:
            values_max = 0
        if values_min and values_max != 0:
            list_of_decr.append((values_max - values_min) / 2 / max_freq_in_range[i])
        elif values_max == 0 and values_min != 0:
            list_of_decr.append((max_freq_in_range[i] - values_min) / max_freq_in_range[i])
        elif values_max != 0 and values_min == 0:
            list_of_decr.append((values_max - max_freq_in_range[i]) / max_freq_in_range[i])
    # calculating mean and MSE of classic decrement for range(start-stop) excluding NaN
    if len(list_of_decr) != 0:
        mean_decr = np.nanmean(list_of_decr)
        mse_decr = np.nanstd(list_of_decr)
    else:
        mean_decr = -1
        mse_decr = -1
    # returning variables
    return mean_decr, mse_decr


def classic_decr_without_np(data, start, stop):
    start = int(start)
    stop = int(stop)
    sqrt_min = 1 / math.sqrt(2) * 0.95
    sqrt_max = 1 / math.sqrt(2) * 1.05
    list_of_tuples = []
    # getting list of tuples with
    # structure ['coord', 'amp_max', 'freq_max', 'freq_min_1', 'freq_min_2', 'found_freq_left', 'found_freq_right']
    for i in range(start, stop):
        try:
            # getting parameters of the maximum with structure
            # ['coord', 'amp_max', 'freq_max', 'freq_min_1', 'freq_min_2']
            values = []
            if data.iloc[0, i] < data.iloc[0, i + 1] > data.iloc[0, i + 2]:
                values = [i + 1, data.iloc[0, i + 1], data.iloc[1, i + 1], sqrt_min * data.iloc[0, i + 1],
                          sqrt_max * data.iloc[0, i + 1]]
                values_np = []
                # getting values at the left side form the maximum
                for j in range(55):
                    try:
                        if values[3] <= data.iloc[0, values[0] - j] <= values[4]:
                            values_np.append(data.iloc[1, i - j])
                    except IndexError:
                        break
                if len(values_np) > 0:
                    values.append(values_np[0])
                else:
                    values.append(0)
                # getting values at the right side form the maximum
                values_np = []
                for j in range(55):
                    try:
                        if values[3] <= data.iloc[0, values[0] + j] <= values[4]:
                            values_np.append(data.iloc[1, i + j])
                    except IndexError:
                        break
                if len(values_np) > 0:
                    values.append(values_np[0])
                else:
                    values.append(0)

            # converting structure of values to ['freq_max', 'found_freq_left', 'found_freq_right']
            values = tuple([values[2], values[5], values[6]])
            # dropping empty tuples
            if len(values) > 0:
                list_of_tuples.append(values)
        except IndexError:
            continue
    # calculating decrement
    list_of_decr = []
    for i in list_of_tuples:
        if i[2] and i[1] != 0:
            d = (i[2] - i[1]) / 2 / i[0]
            list_of_decr.append(d)
        elif i[2] == 0 and i[1] != 0:
            d = (i[0] - i[1]) / i[0]
            list_of_decr.append(d)
        elif i[2] != 0 and i[1] == 0:
            d = (i[2] - i[0]) / i[0]
            list_of_decr.append(d)
    # calculating mean and MSE of classic decrement for range(start-stop) excluding NaN
    if len(list_of_decr) != 0:
        mean_decr = np.nanmean(list_of_decr)
        mse_decr = np.nanstd(list_of_decr)
    else:
        mean_decr = -1
        mse_decr = -1
    # returning variables
    return mean_decr, mse_decr


def calc_with_np():
    columns = ['inter_max', 'inter_min', 'parameter_classic', 'mse_classic']
    df_calc1 = pd.DataFrame(columns=columns)
    df_calc2 = pd.DataFrame(columns=columns)

    for k in range(2, 4):
        start = 10
        for stop in range(11, 200, k):
            amps_in_range = np.array(data.iloc[0, start:stop+1])
            freq_in_range = np.array(data.iloc[1, start:stop+1])
            classic, mse_classic = classic_decr_without_np(data, start, stop)
            classic1, mse_classic1 = classic_decr_with_np(amps_in_range, freq_in_range)
            inter_min = round(float(data.iloc[1, start]), 7)
            inter_max = round(float(data.iloc[1, stop]), 7)
            values1 = [inter_max, inter_min, classic1, mse_classic1]
            values2 = [inter_max, inter_min, classic, mse_classic]
            dict_append1 = [{a: b for a, b in zip(columns, values1)}]
            dict_append2 = [{a: b for a, b in zip(columns, values2)}]
            df_calc1 = df_calc1.append(dict_append1)
            df_calc2 = df_calc2.append(dict_append2)
            start = stop
            c2.append((classic, mse_classic))
            c1.append((classic1, mse_classic1))



def calc_without_np():
    columns = ['inter_max', 'inter_min', 'parameter_classic', 'mse_classic']
    df_calc = pd.DataFrame(columns=columns)
    for k in range(2, 4):
        start = 0
        for stop in range(1, 200, k):
            classic, mse_classic = classic_decr_without_np(data, start, stop)
            inter_min = round(float(data.iloc[1, start]), 7)
            inter_max = round(float(data.iloc[1, stop]), 7)
            values = [inter_max, inter_min, classic, mse_classic]
            dict_append = [{a: b for a, b in zip(columns, values)}]
            df_calc = df_calc.append(dict_append)
            start = stop
            c2.append((classic, mse_classic))


data = pd.read_csv(r'Raw_AFR/СОП-05_100_500_5,0_14.txt', header=None, delimiter='\s+', decimal=',', nrows=1)
data.loc[1] = [100 + i * 400 / 24001 for i in range(len(data.loc[0]))]

c1 = []
c2 = []

print(timeit.timeit(calc_with_np, number=20))

