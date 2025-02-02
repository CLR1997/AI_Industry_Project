import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from core.model import *
from core.generate import *
from core.data import *
from core.metrics import *
import seaborn as sns
sns.set_palette('deep')
sns.set_theme()

name = 'forecasting'
res_dir = f'benchmarks/{name}/results'
agg_res_dir = f'benchmarks/{name}/aggregated_results'
data_dir = f'benchmarks/{name}/data'
prep_dir = f'benchmarks/{name}/preprocess'


#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
wind_in, wind_out = 8, 4
split_perc = 0.8


#######################################################################################################
# Results Extraction
#######################################################################################################
rrd = RegRealistic_Data(inst_by_period=20)
#rrd.get_data(data_dir)
rrd.data = pd.read_csv('benchmarks/forecasting/data/series.csv')selected_series = pd.read_csv(f'{prep_dir}/selected.csv')
ids = selected_series['id'].unique()

model_generators = {}
for ID in ids:
    series = rrd.get_series(ID=ID)
    series, series_train, series_test = rrd.split(series=series, split_perc=split_perc, std=True)
    x_train, y_train = rrd.expand_xy(series=series_train, wind_in=wind_in, wind_out=wind_out)
    x_test, y_test = rrd.expand_xy(series=series_test, wind_in=wind_in, wind_out=wind_out)
    model_generators[ID] = ModelGenerator(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, mode='regression')

results = {}
for model in os.listdir(res_dir):
    log = pickle.load(open(f'{res_dir}/{model}', 'rb'))
    ID = model.split('_')[1]
    log = model_generators[ID].test(log_file=f'{res_dir}/{model}', P=log['P'], p=log['p'], R=log['R'], r=log['r'], print_log=False)
    model = model.split('.')[0]
    results[model] = log


frame = pd.DataFrame({
    'id' : [model.split("_")[1] for model in results],
    'quantile' : [model.split('_')[2] for model in results],
    'model' : [model.split('_')[0] for model in results],
    'layer_type' : [model.split("_")[3] for model in results],
    'depth' : [int(model.split('_')[4]) if model.split('_')[4] != 'none' else 0 for model in results],
    'box' : [model.split('_')[5] for model in results],
    'r2_test' : [results[model]['r2_test'] for model in results],
    'mse_test' : [results[model]['mse_test'] for model in results],
    'slack_tot' : [results[model]['model_test_slack_tot_violation'] for model in results],
    'slack_mean' : [results[model]['model_test_slack_mean_violation'] for model in results],
    'membership' : [results[model]['model_test_membership_violation'] for model in results],
    }, index = list(results.keys()))
frame['quantile'] = frame['quantile'].replace({'100' : '1.00', '95' : '0.95', '90' : '0.90'})
frame['quantile'] = pd.Categorical(frame['quantile'], categories=['1.00', '0.95', '0.90'], ordered=True)
frame = frame.sort_values(['id', 'quantile'])
frame = frame.reset_index().drop('index', axis=1)


####################################################################
# Baseline
####################################################################
baseline = frame[frame['model'].isin(['preprocess', 'postprocess', 'oracle'])]
smle = frame[(frame['model'] == 'smle') & (frame['layer_type'] == 'relu') & (frame['depth'] == 5) & (frame['box'] == 'constant')] 
baseline = pd.concat([baseline, smle], axis = 0)
baseline['model'] = pd.Categorical(baseline['model'], categories=['oracle', 'preprocess', 'postprocess', 'smle'], ordered=True)


metric = 'r2'
ylims = 0.4, 1.01
plt.figure(figsize=(16, 8))
sns.boxplot(data=baseline, x='quantile', y=f'{metric}_test', hue='model', palette='deep')
plt.ylim(ylims[0], ylims[1])
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(title='', loc='lower left', fontsize=22)
plt.savefig(f'{agg_res_dir}/baseline.pdf', bbox_inches='tight')
plt.close()


####################################################################
# Layer Type
####################################################################
parameter = 'layer_type'
smle = frame[(frame['model'] == 'smle') & (frame['depth'] == 5) & (frame['box'] == 'constant')]

smle = smle.groupby(['quantile', parameter])['r2_test'].mean().reset_index()
smle = smle.pivot(index='quantile', columns=parameter)
smle = smle['r2_test'].reset_index()
smle.columns.name = None
table = smle


####################################################################
# Hidden Complexity
####################################################################
parameter = 'depth'
smle = frame[(frame['model'] == 'smle') & (frame['layer_type'] == 'relu') & (frame['box'] == 'constant')]
smle['depth'] = smle['depth'].replace({3 : 'small', 5 : 'large'})

smle = smle.groupby(['quantile', parameter])['r2_test'].mean().reset_index()
smle = smle.pivot(index='quantile', columns=parameter)
smle = smle['r2_test'].reset_index()
smle.columns.name = None
table = pd.merge(table, smle, on='quantile')


####################################################################
# Overapproximator Complexity
####################################################################
parameter = 'box'
smle = frame[(frame['model'] == 'smle') & (frame['layer_type'] == 'relu') & (frame['depth'] == 5)]

smle = smle.groupby(['quantile', parameter])['r2_test'].mean().reset_index()
smle = smle.pivot(index='quantile', columns=parameter)
smle = smle['r2_test'].reset_index()
smle.columns.name = None
table = pd.merge(table, smle, on='quantile')
table = table[['quantile', 'constant', 'linear', 'small', 'large',  'relu', 'lstm']]


####################################################################
# Ablation Study
####################################################################
table = table.melt(id_vars='quantile', var_name='condition', value_name='r2')
table['type'] = table['condition'].map({'constant' : 'auxiliary complexity', 'linear' : 'auxiliary complexity',
                                        'small' : 'embedding size', 'large' : 'embedding size', 
                                        'relu' : 'embedding type', 'lstm' : 'embedding type'})
table.to_csv(f'{agg_res_dir}/ablation_study.csv', index=False)
