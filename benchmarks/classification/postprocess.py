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
import time

name = 'classification'
res_dir = f'benchmarks/{name}/results'
agg_res_dir = f'benchmarks/{name}/aggregated_results'
data_dir = f'benchmarks/{name}/data'


#######################################################################################################
# Experiment Hyperparametes
#######################################################################################################
split_perc = 0.8


#######################################################################################################
# Results Extraction
#######################################################################################################
crd = ClassRealistic_Data()
crd.get_data(data_dir)
ids = crd.get_ids()


model_generators = {}
for ID in ids:
    dataset = crd.get_dataset(ID=ID)
    x, y = crd.get_xy(dataset)
    x, x_train, x_test, y_train, y_test = crd.split(x, y, split_perc=split_perc)
    model_generators[ID] = ModelGenerator(mode='multilabel-classification', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

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
    'accuracy_test' : [results[model]['accuracy_test'] for model in results],
    'slack_tot' : [results[model]['model_test_slack_tot_violation'] for model in results],
    'slack_mean' : [results[model]['model_test_slack_mean_violation'] for model in results],
    'membership' : [results[model]['model_test_membership_violation'] for model in results],
    }, index = list(results.keys()))

frame['quantile'] = frame['quantile'].map({'none' : 'none', '0' : '0.00', '3' : '0.30', '6' : '0.60'})
frame['quantile'] = pd.Categorical(frame['quantile'], categories=['0.00', '0.30', '0.60'], ordered=True)
frame = frame.sort_values(['id', 'quantile'])
frame = frame.reset_index().drop('index', axis=1)


####################################################################
# Baseline
####################################################################
baseline = frame[frame['model'].isin(['preprocess', 'postprocess', 'oracle', 'smle'])]
baseline['model'] = pd.Categorical(baseline['model'], categories=['oracle', 'preprocess', 'postprocess', 'smle'], ordered=True)


plt.figure(figsize=(16, 8))
sns.boxplot(data=baseline, x='quantile', y=f'accuracy_test', hue='model', palette='deep')
ylims = 0.4, 1.01
plt.ylim(ylims[0], ylims[1])
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(title='', loc='lower left', fontsize=22)
plt.savefig(f'{agg_res_dir}/baseline.pdf', bbox_inches='tight')
plt.close()


#####################################################################
# Inference Time
#####################################################################
layer_type = 'relu'
depth = 3
boxes = 'constant'
times = {}
for model in results.keys():
    model_type, ID, _, _, _, _ = model.split('_')
    x_test, y_test = model_generators[ID].x_test, model_generators[ID].y_test
    input_dim, output_dim = x_test.shape[1], y_test.shape[1]
    P, p, R, r = results[model]['P'], results[model]['p'], results[model]['R'], results[model]['r']


    #################################################################################################################
    # Architectures
    #################################################################################################################
    h = [(int(4*width*np.log2(input_dim*output_dim)), layer_type) for width in range(1, int(depth/2)+1)] + \
            [(int(4*width*np.log2(input_dim*output_dim)), 'relu') for width in range(int(depth/2)+1, 0, -1)] 
    h_aux = int(4*np.log2(input_dim*output_dim))


    #################################################################################################################
    # Model Building
    #################################################################################################################
    if model_type == 'preprocess':
        h = EmbeddedNetwork(architecture=h, name='h')
        net = Unsafe(mode='multilabel-classification', h=h, output_dim=output_dim)

    elif model_type == 'postprocess':
        h = EmbeddedNetwork(architecture=h, name='h')
        net = Unsafe(mode='multilabel-classification', h=h, output_dim=output_dim)
        net = ITP(mode='multilabel-classification', unsafe=net, P=P, p=p, R=R, r=r)

    elif model_type == 'smle':
        h = EmbeddedNetwork(architecture=h, name='h')
        h_lower = EmbeddedNetwork(architecture=h_aux, name='h_lower', output_init=-1.)
        h_upper = EmbeddedNetwork(architecture=h_aux, name='h_upper', output_init=1.)
        net = SMLE(mode='multilabel-classification', P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper)

    else: 
        continue


    net(np.array([[0.]*input_dim]))
    net.set_weights(results[model]['final_weights'])

    start = time.time()
    net(x_test)
    end = time.time()

    times[model] = ((end - start)/x_test.shape[0], (end - start))
    pickle.dump(times, open(f'{agg_res_dir}/times.pkl', 'wb'))


times = pickle.load(open(f'{agg_res_dir}/times.pkl', 'rb'))
times = pd.DataFrame({
    'task' : [model.split('_')[1] for model in times],
    'quantile' : [float('0.' + model.split('_')[2]) for model in times],
    'model' : [model.split('_')[0] for model in times],
    'time' : [times[model][0] for model in times],
    }, index = list(times.keys()))

times = times.pivot(index=['task', 'quantile'], columns='model', values='time').reset_index()
times['postprocess'] = times['postprocess']/times['preprocess']
times['smle'] = times['smle']/times['preprocess']
times = times.drop('preprocess', axis=1)
times = times.melt(id_vars=['task', 'quantile'], value_vars=['postprocess', 'smle'],
                    var_name='model', value_name='time')


plt.figure(figsize=(16, 8))
sns.boxplot(data=times, x='quantile', y=f'time', hue='model', palette='deep', log_scale=(None, 2))
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=22)
plt.yticks(ticks=[2**i for i in range(1, 16, 2)], fontsize=22)
plt.legend(title='', loc='upper left', fontsize=22)
plt.savefig(f'{agg_res_dir}/inference_time.pdf', bbox_inches='tight')
plt.close()
