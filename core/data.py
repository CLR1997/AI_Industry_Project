import os
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import arff
from core.optimization import *

class Data:
    def __init__(self, mode):
        self.mode = mode

    def enforce_safety(self, x_train, y_train, P=None, p=None, R=None, r=None):
        OP = OutputProjection(R=R, r=r, mode=self.mode)
        y_train_safe = y_train.copy()

        for k in range(x_train.shape[0]):
            input_membership = np.all(x_train[k] @ P <= p)
            output_membership = np.all(y_train_safe[k] @ R <= r)
            
            if input_membership and not output_membership:
                y_train_safe[k] = OP.project(y_train_safe[k])

        return x_train, y_train_safe


class Synthetic_Data(Data):
    def __init__(self, train_size, test_size, input_dim):
        super().__init__(mode='regression')
        self.train_size = train_size
        self.test_size = test_size
        self.input_dim = input_dim

    def generate(self, seed, task_difficulty, data_split=0.1, std=True, P=None, p=None, R=None, r=None):
        interpolation, extrapolation = 0, 0
        if data_split: 
            # The following parameters guarantee that the area of both the interpolation and the extrapolation sets 
            # is equal to data_split*2**input_dim, where 2**input_dim=total_area
            interpolation = data_split**(1/self.input_dim)
            extrapolation =  1 - (1 - data_split)**(1/self.input_dim)

        np.random.seed(seed)

        # Train Set
        x_train = np.random.uniform(low=-1, high=1, size=(self.train_size, self.input_dim))
        y_train = np.reshape(np.sum(x_train, axis=1), (self.train_size, 1))**np.arange(1, 5)

        # Test Set
        x_test = np.random.uniform(low=-1, high=1, size=(self.test_size, self.input_dim))
        y_test = np.reshape(np.sum(x_test, axis=1), (self.test_size, 1))**np.arange(1, 5)
        
        if std:
            y_train_mean = np.mean(y_train, axis=0)
            y_train_std = np.std(y_train, axis=0)
            y_train = (y_train-y_train_mean)/y_train_std
            y_test = (y_test-y_train_mean)/y_train_std

        interpolation_filter = np.all(x_train > -interpolation, axis=1) & np.all(x_train < interpolation, axis=1)
        extrapolation_filter = np.any(x_train < -(1 - extrapolation), axis=1) | np.any(x_train > (1 - extrapolation), axis=1)
        x_train, y_train = x_train[(~interpolation_filter) & (~extrapolation_filter)], y_train[(~interpolation_filter) & (~extrapolation_filter)]
        
        interpolation_filter = np.all(x_test > -interpolation, axis=1) & np.all(x_test < interpolation, axis=1)
        extrapolation_filter = np.any(x_test < -(1 - extrapolation), axis=1) | np.any(x_test > (1 - extrapolation), axis=1)
        x_test_inter, y_test_inter = x_test[interpolation_filter], y_test[interpolation_filter]
        x_test_extra, y_test_extra = x_test[extrapolation_filter], y_test[extrapolation_filter]
        x_test, y_test = x_test[(~interpolation_filter) & (~extrapolation_filter)], y_test[(~interpolation_filter) & (~extrapolation_filter)]

        if task_difficulty == 'easy':
            y_train, y_test, y_test_inter, y_test_extra = y_train[:,:2], y_test[:,:2], y_test_inter[:,:2], y_test_extra[:,:2]
        elif task_difficulty == 'medium':
            y_train, y_test, y_test_inter, y_test_extra = y_train[:,2:], y_test[:,2:], y_test_inter[:,2:], y_test_extra[:,2:]

        return x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra


class RegRealistic_Data(Data):
    def __init__(self, inst_by_period=20):
        super().__init__(mode='regression')
        self.inst_by_period = inst_by_period

    def _clean_row(self, row):
        non_nan = row.dropna().tolist()
        row = pd.Series(non_nan + [np.nan] * (len(row) - len(non_nan)))

        return row

    def get_data(self, folder):
        self.data = pd.read_csv(f'{folder}/m4_info.csv')
        self.data = self.data[['M4id', 'category', 'Frequency', 'Horizon', 'SP']]
        self.data = self.data.rename({'M4id' : 'id', 'Frequency' : 'frequency', 'Horizon' : 'horizon', 'SP' : 'period'}, axis=1)

        data_train = pd.DataFrame({'id' : []})
        data_test = pd.DataFrame({'id' : []})
        for period in self.data['period'].unique():
            f = pd.read_csv(f'{folder}/{period}-train.csv')
            f = f.rename({'V1' : 'id'}, axis=1)
            data_train = pd.concat([data_train, f], axis=0)
            f = pd.read_csv(f'{folder}/{period}-test.csv')
            f = f.rename({'V1' : 'id'}, axis=1)
            data_test = pd.concat([data_test, f], axis=0)
        self.data = pd.merge(self.data, pd.merge(data_train, data_test, on='id'), on='id')
        self.data['length'] = self.data.notna().sum(axis=1)
        self.data = self.data.sort_values('length', ascending=False).reset_index().drop('index', axis=1)
        self.data = self.data[['id', 'category', 'frequency', 'horizon', 'period', 'length'] + [col for col in self.data.columns if 'V' in col]]
        self.data = self.data.groupby('period').head(self.inst_by_period)
        self.data = self.data.apply(self._clean_row, axis=1).dropna(axis=1, how='all')
        self.data.columns = ['id', 'category', 'frequency', 'horizon', 'period', 'length'] + [f'v{i}' for i in range(len(self.data.columns)-6)]

    def get_ids(self):
        ids = self.data['id'].unique().tolist()

        return ids

    def get_series(self, ID):
        series = self.data.loc[self.data['id'] == ID, [col for col in self.data.columns if 'v' in col]].iloc[0].dropna()

        return series

    def split(self, series, split_perc, std=True):
        series_train = series.iloc[:int(len(series)*split_perc)]
        series_test = series.iloc[int(len(series)*split_perc):]

        if std:
            series_train_mean = np.mean(series_train, axis=0)
            series_train_std = np.std(series_train, axis=0)
            series_train = (series_train-series_train_mean)/series_train_std
            series_test = (series_test-series_train_mean)/series_train_std
            series = (series-series_train_mean)/series_train_std

        return series, series_train, series_test

    def expand(self, series, wind):
        z = [series.iloc[i:len(series)-wind+i+1].values for i in range(0, wind)]
        z = np.stack(z, axis=1)

        return z
    
    def compress(self, z):
        series = z[0, :].tolist() + z[1:, -1].tolist()
        series = pd.Series(series, index=[f'v{i}' for i in range(len(series))])

        return series

    def expand_xy(self, series, wind_in, wind_out):
        x = self.expand(series[:-wind_out], wind_in)
        y = self.expand(series[wind_in:], wind_out)
        
        return x, y
    
    def apply_differencing(self, x, y):
        x_delta = x - np.expand_dims(x[:,-1], 1)
        y_delta = y - np.expand_dims(x[:,-1], 1)

        return x_delta, y_delta

    def reverse_differencing_y(self, x, y_delta):
        y = y_delta + np.expand_dims(x[:,-1], 1)

        return y

    def plot(self, path, series, relation=None, plot_type='scatter'):

        if relation == 'train-test':
            series_train, series_test = series
            series = pd.DataFrame({
                'values' : series_train.tolist() + series_test.tolist(),
                'set' : len(series_train) * ['train'] + len(series_test) * ['test']
                })
            plt.figure(figsize=(10,5))
            if plot_type == 'scatter':
                sns.scatterplot(series, x=series.index, y='values', hue='set')
            elif plot_type == 'line':
                sns.lineplot(series, x=series.index, y='values', hue='set')
            plt.legend().set_title('')
            plt.xlabel('')
            plt.ylabel('')

        elif relation == 'true-pred':
            series_true, series_pred = series
            series = pd.DataFrame({
                'true' : series_true.tolist(),
                'pred' : series_pred.tolist(),
                })
            if plot_type == 'scatter':
                plt.figure(figsize=(10,10))
                sns.scatterplot(series, x='true', y='pred')
                plt.xlabel('true')
                plt.ylabel('pred')
            elif plot_type == 'line':
                plt.figure(figsize=(10,5))
                sns.lineplot(series['true'].values, label='true')
                sns.lineplot(series['pred'].values, label='pred')
                plt.xlabel('')
                plt.ylabel('')
        else:
            plt.figure(figsize=(10,5))
            if plot_type == 'scatter':
                sns.scatterplot(series.values)
            elif plot_type == 'line':
                sns.lineplot(series.values)
            plt.xlabel('')
            plt.ylabel('')
        
        plt.savefig(path)
        plt.close()


class ClassRealistic_Data(Data):
    def __init__(self):
        super().__init__(mode='multilabel-classification')

    def get_data(self, folder):
        self. data = {}
        for name in os.listdir(folder):
            print(f'Processing {name}')
            name = name.split('.arff')[0]
            dataset = arff.load(open(f'{folder}/{name}.arff', 'rt'))
            split = int(dataset['relation'].split(' ')[-1])
            features = [f'feature_{attribute[0]}' for attribute in dataset['attributes'][split:]] if split > 0 else [f'feature_{attribute[0]}' for attribute in dataset['attributes'][:split]]
            labels = [f'label_{attribute[0]}' for attribute in dataset['attributes'][:split]] if split > 0 else [f'label_{attribute[0]}' for attribute in dataset['attributes'][split:]]
            attributes = dataset['attributes']
            dataset = pd.DataFrame(dataset['data'])
            dataset.columns = labels + features if split > 0 else features + labels
            if len(dataset[features].select_dtypes(include=['object']).columns) > 0:
                dataset = pd.concat((dataset, pd.get_dummies(dataset[features].select_dtypes(include=['object']), dtype=float, drop_first=True)), axis=1)
                dataset = dataset.drop(dataset[features].select_dtypes(include=['object']), axis=1)
            dataset[labels] = dataset[[col for col in dataset.columns if 'label_' in col]].astype(int)
            dataset = dataset[[col for col in dataset.columns if 'feature_' in col] + [col for col in dataset.columns if 'label_' in col]]
            self.data[name] = dataset

    def get_ids(self):
        ids = list(self.data.keys())

        return ids

    def get_dataset(self, ID):
        dataset = self.data[ID]

        return dataset

    def get_xy(self, dataset):
        x, y = dataset[[col for col in dataset if 'feature' in col]].values, dataset[[col for col in dataset if 'label' in col]].values 
       
        return x, y

    def split(self, x, y, split_perc, std=True):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_perc, random_state=0)

        if std:
            x_train_mean = np.mean(x_train, axis=0)
            x_train_std = np.std(x_train, axis=0)
            x = (x-x_train_mean)/x_train_std
            x_train = (x_train-x_train_mean)/x_train_std
            x_test = (x_test-x_train_mean)/x_train_std

        return x, x_train, x_test, y_train, y_test
