import pandas as pd
import numpy as np
import re
import os.path
from tqdm import tqdm
import os
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# this function filtering data according to frequency
def filter_freq(data, minfre, maxfre, min_rewr, max_rewr):
    print('Filtering Data')
    # filtering on frequencies
    print('Data was filtered in range ', str(minfre) + '-' + str(maxfre), 'kHz')
    data = rewrite_freq(data, minfre, maxfre)  # writing frequencies for whole range
    print('Rewriting data in range ', str(min_rewr) + '-' + str(max_rewr), 'kHz')
    data = data.loc[:, (data.loc[1] >= min_rewr) & (data.loc[1] <= max_rewr)]
    data.columns = [i for i in range(len(data.loc[0]))]  # reindexing columns
    print('Done!')
    return data


# this function rewriting frequencies
def rewrite_freq(data, min_rewr, max_rewr):
    d1 = int(len(data.loc[0]))
    d2 = round((max_rewr - min_rewr) / d1, 7)
    data.loc[1] = [min_rewr + i * d2 for i in range(d1)]
    return data


#  this function simply counts relief for range start stop
def relief(data, start, stop):
    start = int(start)
    stop = int(stop)
    diff = np.diff(data.loc[0, start:stop].to_list()) / np.diff(data.loc[1, start:stop].to_list())
    rel = round(np.trapz(abs(diff), x=data.loc[1, start + 1:stop]) /
                np.trapz(data.loc[0, start:stop], x=data.loc[1, start:stop]), 5)
    return rel


#  this function simply performs head_method for range start stop
def heads(data, start, stop):
    start = int(start)
    stop = int(stop)
    # columns = ['amp_1', 'amp_2', 'amp_3', 'amp_4', 'amp_5' 'freq_1', 'freq_2', 'freq_3', 'freq_4, freq_5']
    q_list = []
    for i in range(start, stop):
        try:
            if data.iloc[0, i] < data.iloc[0, i + 1] > data.iloc[0, i + 2]:
                values = np.append(data.iloc[0, i - 1: i + 4], data.iloc[1, i - 1: i + 4])
                if len(values) == 10:
                    z = np.polyfit(values[0:4], values[5:9], 2)[0]
                    if z < 0:
                        q = math.sqrt(-(float(values[2])) / z) / (float(values[7]))
                        q_list.append(q)
        except IndexError:
            continue
    # calculating mean and MSE of heads decrement for range(start-stop) excluding NaN
    if len(q_list) != 0:
        q_mean = np.nanmean(q_list)
        q_mse = np.nanstd(q_list)
    else:
        q_mean = -1
        q_mse = -1
    return q_mean, q_mse


#  this function simply performs classic_method for range start stop
def classic_decr(data, start, stop):
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
                    values_np.sort(reverse=True)
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
                    values_np.sort(reverse=False)
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


# class stands for creation data for further transformation
class ProcessingAfr:
    dir = 'Raw_AFR'  # 'АЧХ чистые'
    lis = os.listdir(dir)
    params = {line.split()[0][1:-1]: float(line.split()[1]) for line in open('params.txt', 'r')}
    methods = {'r': 'Processed_AFR_relief',
               'h': 'Processed_AFR_heads',
               's': 'Processed_AFR_straight',
               'all': 'Processed_AFR_all'}

    def __init__(self):
        self.file_name = None
        self.path_correct = None
        self.data = pd.DataFrame
        self.name = None
        self.name_splitted = None
        self.name_sample = None
        self.feature = None
        self.minfre = None
        self.maxfre = None
        self.min_rewr = None
        self.max_rewr = None
        self.df_calc = pd.DataFrame()

    def filter_data(self, file_name, min_rewr, max_rewr):
        self.file_name = file_name
        self.path_correct = ProcessingAfr.dir + '/' + file_name  # creating correct file path
        self.name = os.path.splitext(file_name)[0]
        self.name_splitted = re.split('_', self.name)
        self.name_sample = self.name_splitted[0]
        self.feature = self.params.get(self.name_sample)
        self.minfre = float(self.name_splitted[1])
        self.maxfre = float(self.name_splitted[2])
        self.min_rewr = float(min_rewr)
        self.max_rewr = float(max_rewr)
        self.data = pd.read_csv('{}'.format(self.path_correct), header=None, decimal=",", delimiter=r"\s+", nrows=1)
        print('Loading AFR for: ', self.name_splitted[0] + '_' + self.name_splitted[4])
        self.data = filter_freq(self.data, minfre=self.minfre, maxfre=self.maxfre,
                                min_rewr=self.min_rewr, max_rewr=self.max_rewr)
        return self.data

    def df_calc_params(self):
        data = self.data
        name = self.name_sample
        feature = ProcessingAfr.params.get(name)
        columns = ['name', 'feature', 'inter_max', 'inter_min', 'parameter_rel',
                   'parameter_heads', 'mse_heads', 'parameter_classic', 'mse_classic']
        df_calc = pd.DataFrame(columns=columns)
        for k in tqdm(range(2, len(data.iloc[0])), desc='Counting...', leave=True):
            start = 0
            for stop in range(1, len(data.iloc[0]), k):
                rel = relief(data, start, stop)
                head, mse_head = heads(data, start, stop)
                classic, mse_classic = classic_decr(data, start, stop)
                inter_min = round(float(data.iloc[1, start]), 7)
                inter_max = round(float(data.iloc[1, stop]), 7)
                values = [name, feature, inter_max, inter_min, rel, head, mse_head, classic, mse_classic]
                dict_append = [{a: b for a, b in zip(columns, values)}]
                df_calc = df_calc.append(dict_append)
                start = stop
        self.df_calc = df_calc

    def save_data(self, method_type):
        df_calc = self.df_calc
        destination_folder = ProcessingAfr.methods.get(method_type)
        for_format = {
            'a': destination_folder,
            'b': method_type,
            'c': self.name_sample,
            'd': self.feature,
            'e': self.name_splitted[4]
        }
        print('Saving data...')
        df_calc.to_csv(r'{a}/{b}_{c}_{d}_{e}.csv'.format(**for_format), header=False, index=False)
        return print('File was successfully saved!')


# class stands for averaging data and performing multiple linear regression
class Development:
    dir = 'Processed_AFR_all'
    dir_2 = 'Averaged_AFR'
    lis = os.listdir(dir)
    lis_2 = os.listdir(dir_2)
    groups = {'test': 'SOP', 'real_1': 'MOX', 'real_2': 'MOX_BN'}

    def __init__(self):
        self.data = pd.DataFrame()
        self.subfolder = None
        self.data_set = pd.DataFrame()

    def run(self):
        names = self.lis
        parameters = list(ProcessingAfr.params.keys())
        for item in parameters:
            current_frame = pd.DataFrame()
            for name in names:
                if item in name:
                    correct_path = self.dir + r'/' + str(name)
                    temporary_frame = pd.read_csv(correct_path, header=None)
                    current_frame = pd.concat([current_frame, temporary_frame], axis=1, ignore_index=True)
            del temporary_frame
            current_frame.columns = [i for i in range(len(current_frame.iloc[0]))]
            self.data = current_frame
            del current_frame
            self.averaging()
            self.save_data()

    def averaging(self):
        current_frame = self.data
        beginning = current_frame.iloc[:, 0:4]
        mean_rel = current_frame.iloc[:, 4::9].mean(axis=0)
        std_rel = current_frame.iloc[:, 4::9].std(axis=0)
        mean_head = current_frame.iloc[:, 5::9].mean(axis=0)
        std_head = current_frame.iloc[:, 5::9].std(axis=0)
        mean_head_std = current_frame.iloc[:, 6::9].mean(axis=0)
        std_head_std = current_frame.iloc[:, 6::9].std(axis=0)
        mean_classic = current_frame.iloc[:, 7::9].mean(axis=0)
        std_classic = current_frame.iloc[:, 7::9].std(axis=0)
        mean_classic_std = current_frame.iloc[:, 8::9].mean(axis=0)
        std_classic_std = current_frame.iloc[:, 8::9].std(axis=0)

        list_for_concat = [
            beginning,
            mean_rel, std_rel,
            mean_head, std_head,
            mean_head_std, std_head_std,
            mean_classic, std_classic,
            mean_classic_std, std_classic_std
        ]
        ending = pd.concat(list_for_concat, axis=1, ignore_index=True)
        ending.columns = [i for i in range(len(ending))]
        self.data = ending

    def save_data(self):
        df_calc = self.data
        destination_folder = 'Averaged_AFR'
        method_type = 'averaged'
        name_sample = str(df_calc.iloc[0, 0])
        feature = str(df_calc.iloc[0, 1])
        kind_of = 'tvel'
        for_format = {
            'a': destination_folder,
            'b': method_type,
            'c': name_sample,
            'd': feature,
            'z': kind_of
        }
        print('Saving data...')
        if 'СОП' in for_format['c']:
            for_format['z'] = 'sop'
        else:
            for_format['z'] = 'tvel'
        df_calc.to_csv(r'{a}/{z}/{b}_{c}_{d}.csv'.format(**for_format), header=False, index=False)
        return print('File was successfully saved!')

    def create_dataset(self, subfolder):
        self.subfolder = subfolder
        folder = r'Averaged_AFR/' + str(subfolder)
        lis = os.listdir(folder)
        for name in lis:
            path = folder + name
            loader = pd.read_csv(path, header=None)
            self.data_set = pd.concat([self.data_set, loader], axis=1, ignore_index=True)
        ready_for_reg_path = r'Regression_ready_AFR/' + str(subfolder)
        self.data_set.to_csv(ready_for_reg_path, header=False, index=False)

    def make_prediction(self):
        lm = LinearRegression()
        data_set = self.data_set
        x1 = np.array(data_set.iloc[:, 4::17])
        x2 = np.array(data_set.iloc[:, 8::17])
        x3 = np.array(data_set.iloc[:, 12::17])
        X = [x1, x2, x3]
        y = np.array(data_set.iloc[:, 1::17])
        lm.fit(X, y)
        y_hat = lm.predict(X)
        r2_score(y, y_hat)
        lm.score(X, y)
        coef = lm.coef_
        intercept = lm.intercept_


first = ProcessingAfr()
for i in ProcessingAfr.lis:
    first.filter_data(i, 100, 400)
    first.df_calc_params()
    first.save_data('all')
del first

second = Development()
second.run()
second.create_dataset('sop')
second.create_dataset('tvel')