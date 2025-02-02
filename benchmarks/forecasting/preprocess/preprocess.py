import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from core.data import *
from core.property import *
from core.model import *
from core.generate import *


name = 'forecasting'
directory = f'benchmarks/{name}'
data_dir = f'{directory}/data'
prep_dir = f'{directory}/preprocess'
res_dir = f'{prep_dir}/results'
optimizer = 'adam'
loss = 'mse'
batch_size = 32
epochs = 1000
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
validation_split = 0.2
wind_in, wind_out = 8, 4
split_perc = 0.8
quantiles = [100, 95, 90, 85, 80]
inst_by_period=20


if os.path.isfile(f'{prep_dir}/all.csv'):
    violations = pd.read_csv(f'{prep_dir}/all.csv')
else:
    #################################################################################################################
    # Architectures
    #################################################################################################################
    depth = 3
    h_architecture = [(width*wind_in*wind_out, 'relu') for width in range(1, int(depth/2)+1)] + \
            [(width*wind_in*wind_out, 'relu') for width in range(int(depth/2)+1, 0, -1)]
    h_aux_architecture = [(wind_in*wind_out, 'linear')]

    h = EmbeddedNetwork(architecture=h_architecture)
    h_lower = EmbeddedNetwork(architecture=h_aux_architecture)
    h_upper = EmbeddedNetwork(architecture=h_aux_architecture)


    #################################################################################################################
    # Data
    #################################################################################################################
    rrd = RegRealistic_Data(inst_by_period=20)
    #rrd.get_data(data_dir)
    rrd.data = pd.read_csv('benchmarks/forecasting/data/series.csv')
    ids = rrd.get_ids()


    for ID in ids:
        #################################################################################################################
        # Select Series -- Split -- Compute Windows
        #################################################################################################################
        series = rrd.get_series(ID=ID)
        series, series_train, series_test = rrd.split(series=series, split_perc=split_perc, std=True)

        x_train, y_train = rrd.expand_xy(series=series_train, wind_in=wind_in, wind_out=wind_out)
        x_test, y_test = rrd.expand_xy(series=series_test, wind_in=wind_in, wind_out=wind_out)


        #################################################################################################################
        # Property
        #################################################################################################################
        sp = RegRealistic_Property(wind_in=wind_in, wind_out=wind_out)
        for quantile in quantiles:
            P, p, R, r = sp.generate(series, q=quantile)
            x_train_safe, y_train_safe = rrd.enforce_safety(x_train=x_train, y_train=y_train, P=P, p=p, R=R, r=r)


            #################################################################################################################
            # Training
            #################################################################################################################
            model_generator = ModelGenerator(x_train=x_train_safe, y_train=y_train_safe, 
                                             x_test=x_test, y_test=y_test,
                                             batch_size=batch_size, epochs=epochs, 
                                             optimizer=optimizer, loss=loss, 
                                             validation_split=validation_split, callbacks=[early_stopping], mode='regression')

            log_file = f'{res_dir}/{ID}_{quantile}.pkl'
            if not os.path.isfile(log_file):
                model = model_generator.train(log_file=log_file, model_type='unsafe', h=h, P=P, p=p, R=R, r=r)
                log = model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)


    #################################################################################################################
    # Analysis
    #################################################################################################################
    violations = {}
    for model in os.listdir(res_dir):
        ID, quantile = model.split('.pkl')[0].split('_')
        quantile = int(quantile)
        log = pickle.load(open(f'{res_dir}/{model}', 'rb'))
        series = rrd.get_series(ID=ID)
        series, series_train, series_test = rrd.split(series=series, split_perc=split_perc, std=True)
        x_train, y_train = rrd.expand_xy(series=series_train, wind_in=wind_in, wind_out=wind_out)
        x_test, y_test = rrd.expand_xy(series=series_test, wind_in=wind_in, wind_out=wind_out)
        model_generator = ModelGenerator(x_train=x_train, y_train=y_train, 
                                         x_test=x_test, y_test=y_test,
                                         batch_size=batch_size, epochs=epochs, 
                                         optimizer=optimizer, loss=loss, 
                                         validation_split=validation_split, callbacks=[early_stopping])
        log = model_generator.test(log=f'{res_dir}/{model}', P=log['P'], p=log['p'], R=log['R'], r=log['r'])
        violations[(ID, quantile)] = (
                    log['model_test_slack_tot_violation'],
                    log['model_test_slack_mean_violation'],
                    log['model_test_membership_violation'],
                    log['r2_test'],
                    log['data_test_slack_tot_violation'],
                    log['data_test_slack_mean_violation'],
                    log['data_test_membership_violation'],
                    )

    violations = pd.DataFrame({
        'id' : [ID for ID, quantile in violations.keys()],
        'quantile' : [quantile for ID, quantile in violations.keys()],
        'tot_model' : [violations[key][0] for key in violations.keys()],
        'tot_data' : [violations[key][4] for key in violations.keys()],
        'mean_model' : [violations[key][1] for key in violations.keys()],
        'mean_data' : [violations[key][5] for key in violations.keys()],
        'membership_model' : [violations[key][2] for key in violations.keys()],
        'membership_data' : [violations[key][6] for key in violations.keys()],
        'r2_test' : [violations[key][3] for key in violations.keys()],
        })
    violations = violations.sort_values(['id', 'quantile'], ascending=False)
    violations.to_csv('{prep_dir}/all.csv', index=False)

ids1 = set(violations[violations['quantile'] != 100].groupby('id').filter(lambda x: (x['membership_model'] != 0).any())['id'])
ids2 = set(violations.groupby('id').filter(lambda x: (x['r2_test'] >= 0.8).all())['id'])
ids = ids1 & ids2
violations = violations[violations['id'].isin(ids)]
violations.to_csv(f'{prep_dir}/selected.csv', index=False)
