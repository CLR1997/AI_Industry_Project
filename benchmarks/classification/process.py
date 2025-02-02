import os
import numpy as np
import pandas as pd
import tensorflow as tf
from core.model import *
from core.generate import *
from core.property import *
from core.optimization import *
from core.metrics import *

data_dir = 'benchmarks/classification/data'
res_dir = 'benchmarks/classification/results'


#################################################################################################################
# Hyperpoarameters
#################################################################################################################
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['accuracy']
batch_size = 32
epochs = 1000
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
validation_split = 0.2
split_perc = 0.8
quantiles = [0.0, 0.3, 0.6]
layer_types = ['relu']
depths = [3]
boxes = ['constant']


#################################################################################################################
# Data
#################################################################################################################
crd = ClassRealistic_Data()
crd.get_data(data_dir)
ids = crd.get_ids()


#################################################################################################################
# Process
#################################################################################################################
for ID in ids:
    
    dataset = crd.get_dataset(ID=ID)
    x, y = crd.get_xy(dataset)
    x, x_train, x_test, y_train, y_test = crd.split(x, y, split_perc=split_perc)
    input_dim, output_dim = x.shape[1], y.shape[1]
    

    #################################################################################################################
    # Architectures
    #################################################################################################################
    h_architectures = {
            (depth, layer_type) : [(int(4*width*np.log2(input_dim*output_dim)), layer_type) for width in range(1, int(depth/2)+1)] + \
                    [(int(4*width*np.log2(input_dim*output_dim)), 'relu') for width in range(int(depth/2)+1, 0, -1)] \
            for depth in depths for layer_type in layer_types
            }
    h_aux_architectures = {'constant' : int(4*np.log2(input_dim*output_dim))}


    #################################################################################################################
    # Property-Independent Unsafe Model (to initialize Postprocess)
    #################################################################################################################
    model_generator = ModelGenerator(mode='multilabel-classification', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                     batch_size=batch_size, epochs=epochs,
                                     optimizer=optimizer, loss=loss, metrics=metrics,
                                     validation_split=validation_split, callbacks=[early_stopping])

    quantile, layer_type, depth, box = 'none', 'relu', 3, 'none'
    log_file = f'{res_dir}/unsafe_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
    h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
    unsafe = model_generator.train(log_file=log_file, model_type='unsafe', h=h)
    log = model_generator.test(log_file=log_file)


    #################################################################################################################
    # Property
    #################################################################################################################
    for quantile in quantiles:
        sp = ClassRealistic_Property()
        P, p, R, r = sp.generate(x=x, y=y, q=quantile)
        x_train_safe, y_train_safe = crd.enforce_safety(x_train, y_train, P=P, p=p, R=R, r=r)
        quantile = str(quantile).split('.')[1]

        model_generator_safe = ModelGenerator(mode='multilabel-classification', x_train=x_train_safe, y_train=y_train_safe, x_test=x_test, y_test=y_test,
                                              batch_size=batch_size, epochs=epochs,
                                              optimizer=optimizer, loss=loss,
                                              validation_split=validation_split, callbacks=[early_stopping])


        #################################################################################################################
        # Baseline
        #################################################################################################################
        layer_type, depth, box = 'relu', 3, 'none'
        print('======================================================================================================================')
        print(f'postprocess -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file = f'{res_dir}/postprocess_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not  os.path.isfile(log_file):
            postprocess = model_generator.train(log_file=log_file, model_type='itp', P=P, p=p, R=R, r=r, unsafe=unsafe)
            log = model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)

        layer_type, depth, box = 'relu', 3, 'none'
        print('======================================================================================================================')
        print(f'preprocess -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file = f'{res_dir}/preprocess_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not  os.path.isfile(log_file):
            h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
            preprocess = model_generator_safe.train(log_file=log_file, model_type='unsafe', P=P, p=p, R=R, r=r, h=h)
            log = model_generator_safe.test(log_file=log_file, P=P, p=p, R=R, r=r)

        layer_type, depth, box = 'none', 'none', 'none'
        print('======================================================================================================================')
        print(f'oracle -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file = f'{res_dir}/oracle_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not os.path.isfile(log_file):
            oracle = model_generator.train(log_file=log_file, model_type='oracle', P=P, p=p, R=R, r=r)
            log = model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)


        #################################################################################################################
        # SMLE
        #################################################################################################################
        layer_type, depth, box = 'relu', 3, 'constant'
        print('======================================================================================================================')
        print(f'smle -- id: {ID} -- quantile: {quantile} -- layer_type: {layer_type} --  depth: {depth} -- box: {box}')
        print('======================================================================================================================')
        log_file_unsafe = f'{res_dir}/smleunsafe_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        log_file = f'{res_dir}/smle_{ID}_{quantile}_{layer_type}_{depth}_{box}.pkl'
        if not os.path.isfile(log_file):
            try:
                h = EmbeddedNetwork(architecture=h_architectures[depth, layer_type], name='h')
                h_lower = EmbeddedNetwork(architecture=h_aux_architectures[box], name='h_lower', output_init=-1.)
                h_upper = EmbeddedNetwork(architecture=h_aux_architectures[box], name='h_upper', output_init=1.)
                smleunsafe = model_generator_safe.train(log_file=log_file_unsafe, model_type='smle',
                                                              P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper, safe_train=False)
                log = model_generator_safe.test(log_file=log_file_unsafe, P=P, p=p, R=R, r=r)
                smle = model_generator_safe.train(log_file=log_file, model_type='smle',
                                                        P=P, p=p, R=R, r=r,
                                                        h=smleunsafe.h, h_lower=smleunsafe.h_lower, h_upper=smleunsafe.h_upper,
                                                        g=smleunsafe.g, g_poly=smleunsafe.g_poly)
                log = model_generator_safe.test(log_file=log_file, P=P, p=p, R=R, r=r)
            except:
                pass
