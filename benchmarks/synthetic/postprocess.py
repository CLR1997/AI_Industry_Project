import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from core.model import *
from core.generate import *
import seaborn as sns
sns.set_palette('deep')
sns.set_theme()



#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
name = 'synthetic'
res_dir = f'benchmarks/{name}/results'
agg_res_dir = f'benchmarks/{name}/aggregated_results'
data_dir = f'benchmarks/{name}/data'
input_dims = [2, 4, 8]
task_difficulties = ['easy', 'medium', 'hard']


#######################################################################################################
# Results Extraction
#######################################################################################################
data = {}
for dataset in os.listdir(data_dir):
    d = pickle.load(open(f'{data_dir}/{dataset}', 'rb'))
    dataset = dataset.split('.')[0]
    data[dataset] = d


model_generators = {}
for input_dim in input_dims:
    for task_difficulty in task_difficulties:
        task = f'{input_dim}_{task_difficulty}'
        model_generators[input_dim, task_difficulty] = ModelGenerator(
                mode='regression',
                x_train=data[f'x_train_{input_dim}_{task_difficulty}'], y_train=data[f'y_train_{input_dim}_{task_difficulty}'], 
                x_test=data[f'x_test_{input_dim}_{task_difficulty}'], y_test=data[f'y_test_{input_dim}_{task_difficulty}'], 
                x_test_inter=data[f'x_test_inter_{input_dim}_{task_difficulty}'], y_test_inter=data[f'y_test_inter_{input_dim}_{task_difficulty}'], 
                x_test_extra=data[f'x_test_extra_{input_dim}_{task_difficulty}'], y_test_extra=data[f'y_test_extra_{input_dim}_{task_difficulty}'], 
                )


results = {}
for model in os.listdir(res_dir):
    log = pickle.load(open(f'{res_dir}/{model}', 'rb'))
    input_dim, task_difficulty, depth = int(model.split('_')[1]), model.split('_')[2], int(model.split('_')[5])
    if depth != 1:
        log = model_generators[input_dim, task_difficulty].test(log_file=f'{res_dir}/{model}', P=log['P'], p=log['p'], R=log['R'], r=log['r'], print_log=False)
        model = model.split('.')[0]
        results[model] = log


frame = pd.DataFrame({
    'model' : [model.split('_')[0] for model in results],
    'input_dim' : [int(model.split("_")[1]) for model in results],
    'task_difficulty' : [model.split('_')[2] for model in results],
    'property' : [f'{model.split("_")[3]}-{model.split("_")[4]}' for model in results],
    'depth' : [int(model.split('_')[5]) for model in results],
    'box' : [model.split('_')[6] for model in results],
    'data_safety' : [model.split('_')[7] for model in results],
    'train_start' : [model.split('_')[8] for model in results],
    'r2_test' : [results[model]['r2_test'] for model in results],
    'r2_test_inter' : [results[model]['r2_test_inter'] for model in results],
    'r2_test_extra' : [results[model]['r2_test_extra'] for model in results],
    'mse_test' : [results[model]['mse_test'] for model in results],
    'mse_test_inter' : [results[model]['mse_test_inter'] for model in results],
    'mse_test_extra' : [results[model]['mse_test_extra'] for model in results],
    'test_slack_tot_violation' : [results[model]['model_test_slack_tot_violation'] for model in results],
    'test_inter_slack_tot_violation' : [results[model]['model_test_inter_slack_tot_violation'] for model in results],
    'test_extra_slack_tot_violation' : [results[model]['model_test_extra_slack_tot_violation'] for model in results],
    'test_slack_mean_violation' : [results[model]['model_test_slack_mean_violation'] for model in results],
    'test_inter_slack_mean_violation' : [results[model]['model_test_inter_slack_mean_violation'] for model in results],
    'test_extra_slack_mean_violation' : [results[model]['model_test_extra_slack_mean_violation'] for model in results],
    'test_membership_violation' : [results[model]['model_test_membership_violation'] for model in results],
    'test_inter_membership_violation' : [results[model]['model_test_inter_membership_violation'] for model in results],
    'test_extra_membership_violation' : [results[model]['model_test_extra_membership_violation'] for model in results],
    }, index = list(results.keys()))


frame['model'] = frame['model'].replace({'unsafe' : 'preprocess', 'itp' : 'postprocess'})
frame['property'] = frame['input_dim'].astype(str) + '_' + frame['task_difficulty'] + '_' + frame['property']
prop_order = []
for input_dim in input_dims:
    for task_difficulty in task_difficulties:
        f = frame[(frame['model'] == 'oracle') & (frame['input_dim'] == input_dim) & (frame['task_difficulty'] == task_difficulty)]
        prop_order += f[f['model'] == 'oracle'].sort_values(by=f'r2_test', ascending=False)['property'].tolist()
frame['property'] = pd.Categorical(frame['property'], categories=prop_order, ordered=True)
frame = frame.sort_values('property')
frame['property'] = frame['property'].apply(lambda x: x.split('_')[-1])
frame['model'] = pd.Categorical(frame['model'], categories=['oracle', 'preprocess', 'postprocess', 'smle'], ordered=True)
frame['box'] = pd.Categorical(frame['box'], categories=['constant', 'linear'], ordered=True)
frame['data_safety'] = pd.Categorical(frame['data_safety'], categories=['unsafe', 'safe'], ordered=True)
frame['train_start'] = pd.Categorical(frame['train_start'], categories=['none', 'unsafe', 'smle'], ordered=True)

for input_dim in input_dims:
    for task_difficulty in task_difficulties:
        cond = (frame['model'] == 'oracle') & (frame['input_dim'] == input_dim) & (frame['task_difficulty'] == task_difficulty)
        properties = frame.loc[cond, 'property'].tolist()
        properties = {k : f'P{i+1}' for i,k in enumerate(properties)}
        cond = (frame['input_dim'] == input_dim) & (frame['task_difficulty'] == task_difficulty)
        frame.loc[cond, 'property'] = frame.loc[cond, 'property'].replace(properties)


####################################################################
# Baseline
####################################################################
baseline = frame[(frame['model'] == 'preprocess') | (frame['model'] == 'oracle') | (frame['model'] == 'postprocess')]
smle = frame[(frame['model'] == 'smle') & (frame['depth'] == 3) & (frame['box'] == 'linear') & (frame['data_safety'] == 'safe') & (frame['train_start'] == 'none')] 
baseline = pd.concat([baseline, smle], axis = 0)
baseline = baseline[['model', 'input_dim', 'task_difficulty', 'property',
                     'r2_test', 'r2_test_inter', 'r2_test_extra',
                     'mse_test', 'mse_test_inter', 'mse_test_extra']]

metric = 'r2'
test_set = 'test'
ylims = 0.4, 1.01
plt.figure(figsize=(16,8))
sns.boxplot(data=baseline, x='property', y=f'{metric}_{test_set}', hue='model', palette='deep')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('')
plt.ylabel('')
plt.legend(title='', loc='lower left', fontsize=22)
plt.ylim(ylims)
plt.savefig(f'{agg_res_dir}/baseline.pdf', bbox_inches='tight')
plt.close()


####################################################################
# Hidden Complexity
####################################################################
parameter = 'depth'
smle = frame[(frame['model'] == 'smle') & (frame['box'] == 'linear') & (frame['data_safety'] == 'unsafe') & (frame['train_start'] == 'none')] 
smle['depth'] = smle['depth'].replace({3 : 'small', 5 : 'large'})

smle = smle.groupby(['property', parameter])['r2_test'].mean().reset_index()
smle = smle.pivot(index='property', columns=parameter)
smle = smle['r2_test'].reset_index()
smle.columns.name = None
table = smle


####################################################################
# Overapproximator Complexity
####################################################################
parameter = 'box'
smle = frame[(frame['model'] == 'smle') & (frame['depth'] == 3) & (frame['data_safety'] == 'unsafe') & (frame['train_start'] == 'none')] 

smle = smle.groupby(['property', parameter])['r2_test'].mean().reset_index()
smle = smle.pivot(index='property', columns=parameter)
smle = smle['r2_test'].reset_index()
smle.columns.name = None
table = pd.merge(table, smle, on='property')
table = table[['property', 'constant', 'linear', 'small', 'large']]


####################################################################
# Ablation Study
####################################################################
table = table.melt(id_vars='property', var_name='condition', value_name='r2')
table['type'] = table['condition'].map({'constant' : 'auxiliary complexity', 'linear' : 'auxiliary complexity',
                                         'small' : 'embedding size', 'large' : 'embedding size'})
table.to_csv(f'{agg_res_dir}/ablation_study.csv', index=False)



####################################################################
# UNSAFE Slack Violation
####################################################################
slack_tot = frame[frame['model'] == 'preprocess'].rename(columns={
                                                    'test_slack_tot_violation' :'test', 
                                                    'test_inter_slack_tot_violation' : 'test_inter',
                                                    'test_extra_slack_tot_violation' : 'test_extra',
                                                    })
slack_mean = frame[frame['model'] == 'preprocess'].rename(columns={
                                                    'test_slack_mean_violation' :'test', 
                                                    'test_inter_slack_mean_violation' : 'test_inter',
                                                    'test_extra_slack_mean_violation' : 'test_extra',
                                                    })
membership = frame[frame['model'] == 'preprocess'].rename(columns={
                                                    'test_membership_violation' : 'test', 
                                                    'test_inter_membership_violation' : 'test_inter',
                                                    'test_extra_membership_violation' : 'test_extra'
                                                    })
slack_tot = pd.melt(slack_tot, id_vars=['input_dim', 'task_difficulty', 'property'],
                         value_vars=['test', 'test_inter', 'test_extra'],
                         var_name='set', value_name='slack_tot_violation')
slack_mean = pd.melt(slack_mean, id_vars=['input_dim', 'task_difficulty', 'property'],
                         value_vars=['test', 'test_inter', 'test_extra'],
                         var_name='set', value_name='slack_mean_violation')
membership = pd.melt(membership, id_vars=['input_dim', 'task_difficulty', 'property'],
                         value_vars=['test', 'test_inter', 'test_extra'],
                         var_name='set', value_name='membership_violation')
violation = pd.merge(pd.merge(slack_tot, slack_mean, on=['input_dim', 'task_difficulty', 'property', 'set']), membership)

membership = violation[['input_dim', 'task_difficulty', 'property', 'set', 'membership_violation']]
ylims = 0.0, 0.60
plt.figure(figsize=(16,8))
sns.boxplot(data=membership, x='property', y='membership_violation')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22, ticks=np.linspace(0.0, 0.60, 7), labels=['0%', '10%', '20%', '30%', '40%', '50%', '60%'])
plt.xlabel('')
plt.ylabel('')
plt.ylim(ylims)
plt.savefig(f'{agg_res_dir}/preprocess_violation.pdf', bbox_inches='tight')
plt.close()
