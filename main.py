import math
import os
import os.path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# this function filtering data according to frequency
def filter_freq(data, minfre, maxfre, min_rewr, max_rewr):
    print('Filtering Data')
    # filtering on frequencies
    print('Data was loaded in range ', str(minfre) + '-' + str(maxfre), 'kHz')
    data = rewrite_freq(data, minfre, maxfre)  # writing frequencies for whole range
    print('Rewriting data in range ', str(min_rewr) + '-' + str(max_rewr), 'kHz')
    data = data.loc[:, (data.loc[1] >= min_rewr) & (data.loc[1] <= max_rewr)]
    data.columns = [i for i in range(len(data.loc[0]))]  # reindexing columns
    d1 = int(len(data.loc[0]))
    d2 = round((max_rewr - min_rewr) / d1, 7)
    li = [min_rewr + i * d2 for i in range(d1)]
    data = pd.DataFrame([data.loc[0].to_list(), li])
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
                    z = np.polyfit(values[0:5], values[5:10], 2)[0]
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
                for j in range(25):
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
                for j in range(25):
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


# class stands for creation data for further transformation
class ProcessingAfr:
    dir_1 = 'Raw_AFR'  # 'АЧХ чистые'
    lis = os.listdir(dir_1)
    params = dict

    types = ['sop', 'tvel_bn', 'tvel_mox']

    methods = {'r': 'Processed_AFR_relief',
               'h': 'Processed_AFR_heads',
               's': 'Processed_AFR_straight',
               'all': 'Processed_AFR_all'}

    def __init__(self):
        self.data = pd.DataFrame()
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
        self.sop_dict = {line.split()[0][1:-1]: float(line.split()[1])
                         for line in open('sops.txt', 'r', encoding='utf-8')}
        self.tvel_bn_dict = {line.split()[0][1:-1]: float(line.split()[1])
                             for line in open('tvel_bn.txt', 'r', encoding='utf-8')}
        self.tvel_mox_dict = {line.split()[0][1:-1]: float(line.split()[1])
                              for line in open('tvel_mox.txt', 'r', encoding='utf-8')}

    def filter_data(self, file_name, min_rewr, max_rewr):
        self.file_name = file_name
        self.path_correct = ProcessingAfr.dir_1 + '/' + file_name  # creating correct file path
        self.name = os.path.splitext(file_name)[0]
        self.name_splitted = re.split('_', self.name)
        self.name_sample = self.name_splitted[0]
        self.sop_dict.update(self.tvel_bn_dict)
        self.sop_dict.update(self.tvel_mox_dict)
        ProcessingAfr.params = self.sop_dict
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
        feature = ProcessingAfr.params[name]
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
        destination_folder = ProcessingAfr.methods[method_type]
        for_format = {
            'a': destination_folder,
            'b': method_type,
            'c': self.name_sample,
            'd': self.feature,
            'e': self.name_splitted[4]
        }
        print('Saving data...')
        df_calc.to_csv(r'{a}/Methods_{b}_{c}_{d}_{e}.csv'.format(**for_format), header=False, index=False)
        return print('File was successfully saved!')


# class stands for averaging data and performing multiple linear regression_results_of_averaged
class Development:
    dir_1 = 'Processed_AFR_all'
    dir_2 = 'Averaged_AFR'
    lis = os.listdir(dir_1)
    lis_2 = os.listdir(dir_2)

    def __init__(self):
        self.data = pd.DataFrame()
        self.subfolder = None
        self.data_set = pd.DataFrame()
        self.lm_totals = []
        self.kind_of_sample = {
            'sop': {line.split()[0][1:-1]: float(line.split()[1]) for
                    line in open('sops.txt', 'r', encoding='utf-8')},
            'tvel_bn': {line.split()[0][1:-1]: float(line.split()[1]) for
                        line in open('tvel_bn.txt', 'r', encoding='utf-8')},
            'tvel_mox': {line.split()[0][1:-1]: float(line.split()[1]) for
                         line in open('tvel_mox.txt', 'r', encoding='utf-8')}
        }
        self.sample_type = None

    def avg(self, test_type):
        names = self.lis
        test_type = str(test_type)
        self.sample_type = test_type
        parameters = list(self.kind_of_sample.get(test_type).keys())
        current_frame = pd.DataFrame()
        for item in parameters:
            print('Loading data for:' + str(item))
            for name in names:
                if item in name:
                    correct_path = self.dir_1 + r'/' + str(name)
                    temporary_frame = pd.read_csv(correct_path, header=None)
                    current_frame = pd.concat([current_frame, temporary_frame], axis=1, ignore_index=True)
                    del temporary_frame
            if len(current_frame) > 0:
                current_frame.columns = [i for i in range(len(current_frame.iloc[0]))]
                self.data = current_frame
                del current_frame
                self.averaging()
                self.save_data()

    def averaging(self):
        current_frame = self.data
        print('Averaging data for:' + str(current_frame.iloc[0, 0]))
        beginning = current_frame.iloc[:, 0:4]

        mean_rel_exc = (current_frame.iloc[:,
                        range(4, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        std_rel_exc = (current_frame.iloc[:,
                       range(4, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        mean_head_exc = (current_frame.iloc[:,
                         range(5, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        std_head_exc = (current_frame.iloc[:,
                        range(5, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        mean_head_std_exc = (current_frame.iloc[:,
                             range(6, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        std_head_std_exc = (current_frame.iloc[:,
                            range(6, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        mean_classic_exc = (current_frame.iloc[:,
                            range(7, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        std_classic_exc = (current_frame.iloc[:,
                           range(7, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        mean_classic_std_exc = (current_frame.iloc[:,
                                range(8, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2
        std_classic_std_exc = (current_frame.iloc[:,
                               range(8, len(current_frame.iloc[0]), 9)].values != -1).sum(axis=1) < 2

        current_frame.loc[mean_rel_exc, 4::9] = -1
        current_frame.loc[std_rel_exc, 4::9] = -1
        current_frame.loc[mean_head_exc, 5::9] = -1
        current_frame.loc[std_head_exc, 5::9] = -1
        current_frame.loc[mean_head_std_exc, 6::9] = -1
        current_frame.loc[std_head_std_exc, 6::9] = -1
        current_frame.loc[mean_classic_exc, 7::9] = -1
        current_frame.loc[std_classic_exc, 7::9] = -1
        current_frame.loc[mean_classic_std_exc, 8::9] = -1
        current_frame.loc[std_classic_std_exc, 8::9] = -1

        del mean_rel_exc, std_rel_exc, mean_head_exc, std_head_exc, mean_head_std_exc, std_head_std_exc, \
            mean_classic_exc, std_classic_exc, mean_classic_std_exc, std_classic_std_exc

        mean_rel = current_frame.iloc[:, 4::9].mean(axis=1)
        std_rel = current_frame.iloc[:, 4::9].std(axis=1)
        mean_head = current_frame.iloc[:, 5::9].mean(axis=1)
        std_head = current_frame.iloc[:, 5::9].std(axis=1)
        mean_head_std = current_frame.iloc[:, 6::9].mean(axis=1)
        std_head_std = current_frame.iloc[:, 6::9].std(axis=1)
        mean_classic = current_frame.iloc[:, 7::9].mean(axis=1)
        std_classic = current_frame.iloc[:, 7::9].std(axis=1)
        mean_classic_std = current_frame.iloc[:, 8::9].mean(axis=1)
        std_classic_std = current_frame.iloc[:, 8::9].std(axis=1)

        list_for_concat = [
            beginning,
            mean_rel, std_rel,
            mean_head, std_head,
            mean_head_std, std_head_std,
            mean_classic, std_classic,
            mean_classic_std, std_classic_std
        ]
        ending = pd.concat(list_for_concat, axis=1, ignore_index=True)
        del beginning, mean_rel, std_rel, mean_head, std_head, mean_head_std, std_head_std, mean_classic, \
            std_classic, mean_classic_std, std_classic_std, list_for_concat
        ending.columns = [i for i in range(len(ending.iloc[0]))]
        self.data = ending

    def save_data(self):
        df_calc = self.data
        destination_folder = 'Averaged_AFR'
        method_type = 'averaged'
        name_sample = str(df_calc.iloc[0, 0])
        feature = str(df_calc.iloc[0, 1])
        for_format = {
            'a': destination_folder,
            'b': method_type,
            'c': name_sample,
            'd': feature,
            'z': None
        }
        print('Saving data...')
        if self.sample_type == 'sop':
            for_format['z'] = 'sop'
        elif self.sample_type == 'tvel_mox':
            for_format['z'] = 'tvel_mox'
        else:
            for_format['z'] = 'tvel_bn'

        df_calc.to_csv(r'{a}/{z}/{b}_{c}_{d}.csv'.format(**for_format), header=False, index=False)
        self.data = pd.DataFrame()
        self.data_set = pd.DataFrame()
        print('File was successfully saved!')

    def create_averaged_frame(self):
        self.subfolder = self.sample_type
        folder = r'Averaged_AFR/' + str(self.subfolder) + '/'
        lis = os.listdir(folder)
        print('Creating total dataset')
        for name in lis:
            path = folder + name
            loader = pd.read_csv(path, header=None)
            self.data_set = pd.concat([self.data_set, loader], axis=1, ignore_index=True)
        ready_for_reg_path = r'Regression_ready_AFR/' + str(self.subfolder) + r'/averaged_dataset_ready.csv'
        self.drop_minusone()
        self.data_set.to_csv(ready_for_reg_path, header=False, index=False)

    def drop_minusone(self):
        data_set = self.data_set
        data_set.drop(data_set[data_set.values == -1].index, axis=0, inplace=True)
        data_set.index = [k for k in range(len(data_set))]
        self.data_set = data_set

    def multiple_regression(self, kind):

        # you shouldnt use this method now. It is pasted here for future development

        names = self.lis  # list of all non-averaged samples
        keys = list(self.kind_of_sample[kind].keys())
        list_of_files = []
        for num, val in enumerate(names):
            for p in keys:
                if p in val:
                    list_of_files.append(num)

        constant_frame = pd.DataFrame()
        if len(list_of_files) > 0:
            names = list(np.array(names)[list_of_files])
        for name in names:
            correct_path = self.dir_1 + r'/' + str(name)
            temporary_frame = pd.read_csv(correct_path, header=None)
            constant_frame = pd.concat([constant_frame, temporary_frame], axis=1, ignore_index=True)
        del temporary_frame

        dropper = [nu for nu, va in enumerate(constant_frame.values) if -1 in va]
        constant_frame.drop(dropper, axis=0, inplace=True)
        constant_frame.index = [i for i in range(len(constant_frame))]

        column = ['max_int', 'min_int', 'A_rel', 'B_heads', 'C_classic', 'D_intercept', 'r2_score', 'mse']
        regression_frame = pd.DataFrame(data=None, columns=column)
        for j in range(len(constant_frame)):
            x_rel = np.array(constant_frame.iloc[j, 4::9])
            x_heads = np.array(constant_frame.iloc[j, 5::9])
            x_classic = np.array(constant_frame.iloc[j, 7::9])
            x = np.array([x_rel, x_heads, x_classic]).T
            y = np.array(constant_frame.iloc[j, 1::9])
            lm = LinearRegression()
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
            lm.fit(x_train, y_train)
            y_hat = lm.predict(x_test)
            coef = list(lm.coef_)
            intercept = lm.intercept_
            score = r2_score(y_test, y_hat)
            mse = mean_squared_error(y_test, y_hat)
            parameters = pd.DataFrame([(constant_frame.iloc[j, 2], constant_frame.iloc[j, 3], coef[0],
                                        coef[1], coef[2], intercept, score, mse)], columns=column)
            regression_frame = pd.concat([regression_frame, parameters], axis=0, ignore_index=True)

        path = r'Regression_ready_AFR/regression_results_non_averaged/' + \
               self.subfolder + '/t_regression_results_' + self.subfolder + '.csv '
        regression_frame.to_csv(path)
        print('Done!')

    def regress(self, kind, obj, drop):
        path = 'pickle/data/'
        if kind == 'rel':
            column = ['max_int', 'min_int', 'A_rel', 'D_intercept', 'r2_score', 'mse']
            point = int(4)
        elif kind == 'heads':
            column = ['max_int', 'min_int', 'B_heads', 'D_intercept', 'r2_score', 'mse']
            point = int(5)
        elif kind == 'classic':
            column = ['max_int', 'min_int', 'C_classic', 'D_intercept', 'r2_score', 'mse']
            point = int(7)
        else:
            print('Неверное имя метода расчета. Доступны имена rel, heads, classic')
            raise

        names = self.lis  # list of all non-averaged samples
        keys = list(self.kind_of_sample[obj].keys())
        list_of_files = []
        for num, val in enumerate(names):
            for p in keys:
                if p in val:
                    list_of_files.append(num)

        constant_frame = pd.DataFrame()
        if len(list_of_files) > 0:
            names = list(np.array(names)[list_of_files])
        for name in names:
            correct_path = self.dir_1 + r'/' + str(name)
            temporary_frame = pd.read_csv(correct_path, header=None)
            constant_frame = pd.concat([constant_frame, temporary_frame], axis=1, ignore_index=True)
        del temporary_frame
        with open('_'.join([path, obj, kind, 'frame', '.pickle']), 'wb') as f:
            pickle.dump(constant_frame, f)

        if drop == 'y':
            dropper = [nu for nu, va in enumerate(constant_frame.values) if -1 in va]
            constant_frame.drop(dropper, axis=0, inplace=True)
            constant_frame.index = [i for i in range(len(constant_frame))]

        def regr_train_test(x, y):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
            lm.fit(x_train, y_train)
            y_hat = lm.predict(x_test)
            coef = lm.coef_[0]
            intercept = lm.intercept_
            score = r2_score(y_test, y_hat)
            mse = mean_squared_error(y_test, y_hat)
            return coef, intercept, score, mse

        def nonsplit_regr(x, y):
            lm.fit(x, y)
            coef = lm.coef_[0]
            intercept = lm.intercept_
            score = lm.score(x, y)
            mse = mean_squared_error(y, lm.predict(x))
            return coef, intercept, score, mse

        regression_frame = pd.DataFrame(data=None, columns=column)

        for j in range(len(constant_frame)):
            x = np.array(constant_frame.iloc[j, point::9]).reshape(-1, 1)
            y = np.array(constant_frame.iloc[j, 1::9])
            lm = LinearRegression()
            if obj == 'sop':
                coef, intercept, score, mse = regr_train_test(x, y)
            elif obj == 'tvel_mox':
                coef, intercept, score, mse = nonsplit_regr(x, y)
            elif obj == 'tvel_bn':
                coef, intercept, score, mse = regr_train_test(x, y)
            else:
                print('Неверно выбрано имя ТВЭЛа! Доступные именя tvel_bn, tvel_mox, sop ')
                raise

            parameters = pd.DataFrame([(constant_frame.iloc[j, 2], constant_frame.iloc[j, 3], coef,
                                        intercept, score, mse)], columns=column)
            regression_frame = pd.concat([regression_frame, parameters], axis=0, ignore_index=True)

        with open('_'.join([path, obj, kind, 'regr', '.pickle']), 'wb') as f:
            pickle.dump(regression_frame, f)
        # path = r'Regression_ready_AFR/regression_results_non_averaged/' + \
        #       obj + '/t_regression_results_six_samples' + kind + '.csv '
        # regression_frame.to_csv(path)

def calc(data_name):
    first = ProcessingAfr()
    first.filter_data(data_name, 100, 400)
    first.df_calc_params()
    first.save_data('all')
    del first


'''
if __name__ == '__main__':
    with Pool(processes=4) as pool:
        multiple_results = []
        for i in ProcessingAfr.lis:
            multiple_results.append(pool.apply_async(calc, (i,)))
        [res.get() for res in multiple_results]

    second = Development()
    second.avg('sop')
    second.create_averaged_frame()
    second.multiple_regression('sop')
    del second

    second = Development()
    second.avg('tvel_bn')
    second.create_averaged_frame()
    second.regress('r', 'tvel_bn', 'y')
    del second
'''

for obraz in ['sop']:  # , 'tvel_mox', 'tvel_bn']:
    for metod in ['rel', 'classic', 'heads']:
        second = Development()
        second.regress(metod, obraz, 'n')

#
