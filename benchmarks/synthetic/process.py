import os
import pickle
import numpy as np
from core.data import *
from core.model import *
from core.property import *
from core.generate import *


#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
name = 'synthetic'
res_dir = f'benchmarks/{name}/results'
prep_dir = f'benchmarks/{name}/preprocess'

##########################################
# Training
##########################################
optimizer = 'adam'
loss = 'mse'
batch_size = 128
epochs = 1000 
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
validation_split = 0.2
data_seed = 0
train_size = 5000
test_size = 1000
 
##########################################
# tasks
##########################################
tasks = [(2,'easy'), (2, 'medium'), (2,'hard'), (4,'easy'), (4, 'medium'), (4,'hard'), (8,'easy'), (8, 'medium'), (8,'hard')]

##########################################
# properties 
##########################################
print('=========================================================================================================================================================================================')
print(f'properties')
print('=========================================================================================================================================================================================')
n_properties = 6
properties_list = pickle.load(open(f'{prep_dir}/properties.pkl', 'rb'))

properties = {}
for input_dim, task_difficulty in tasks:
    output_constrs = int(np.log2(4) + 2) if task_difficulty == 'hard' else int(np.log2(2) + 2)
    difficulty = properties_list[input_dim, task_difficulty]
    difficulty = difficulty[difficulty['difficulty'] <= 0.5]
    difficulty = np.sort(difficulty, order='difficulty')
    select = [np.abs(difficulty['difficulty'] - i*np.max(difficulty['difficulty'])/(n_properties-1)).argmin() for i in range(n_properties)]
    properties[input_dim, task_difficulty] = np.sort(np.unique(difficulty[select]), order='difficulty')
    if len(properties[input_dim, task_difficulty]) < n_properties:
        select = np.linspace(0, len(difficulty) - 1, n_properties, endpoint=True, dtype=int)
        properties[input_dim, task_difficulty] = np.sort(np.unique(difficulty[select]), order='difficulty')
    print('\n#######################')
    print(f'{input_dim} -- {task_difficulty}')
    print('#######################')
    print(properties[input_dim, task_difficulty])
    properties[input_dim, task_difficulty] = properties[input_dim, task_difficulty]['seeds']


##########################################
# Data Dumping
##########################################
for input_dim, task_difficulty in tasks:
    data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
    x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty)

    pickle.dump(x_train, open(f'benchmarks/{name}/data/x_train_{input_dim}_{task_difficulty}.pkl', 'wb'))
    pickle.dump(x_test, open(f'benchmarks/{name}/data/x_test_{input_dim}_{task_difficulty}.pkl', 'wb'))
    pickle.dump(x_test_inter, open(f'benchmarks/{name}/data/x_test_inter_{input_dim}_{task_difficulty}.pkl', 'wb'))
    pickle.dump(x_test_extra, open(f'benchmarks/{name}/data/x_test_extra_{input_dim}_{task_difficulty}.pkl', 'wb'))

    pickle.dump(y_train, open(f'benchmarks/{name}/data/y_train_{input_dim}_{task_difficulty}.pkl', 'wb'))
    pickle.dump(y_test, open(f'benchmarks/{name}/data/y_test_{input_dim}_{task_difficulty}.pkl', 'wb'))
    pickle.dump(y_test_inter, open(f'benchmarks/{name}/data/y_test_inter_{input_dim}_{task_difficulty}.pkl', 'wb'))
    pickle.dump(y_test_extra, open(f'benchmarks/{name}/data/y_test_extra_{input_dim}_{task_difficulty}.pkl', 'wb'))
 

##########################################
# Architectures
##########################################
h_architectures = {(input_dim, output_dim, depth) : [(width*input_dim*output_dim, 'relu') for width in range(1, int(depth/2)+1)] + [(width*input_dim*output_dim, 'relu') for width in range(int(depth/2)+1, 0, -1)] for input_dim in [2,4,8] for output_dim in [2,4] for depth in [1,3,5]}
h_aux_architectures = {
        **{(input_dim, output_dim, 'constant') : input_dim*output_dim for input_dim in [2,4,8] for output_dim in [2,4]},
        **{(input_dim, output_dim, 'linear') : [(input_dim*output_dim, 'linear')] for input_dim in [2,4,8] for output_dim in [2,4]},
        **{(input_dim, output_dim, 'non-linear') : [(input_dim*output_dim, 'linear'), (int(np.log2(input_dim*output_dim)), 'relu'), (input_dim*output_dim, 'linear')] for input_dim in [2,4,8] for output_dim in [2,4]}
        }


#######################################################################################################
# Baseline
#######################################################################################################
##########################################
# PREPROCESS
##########################################
depth = 3
box = 'none'
data_safety = 'safe'
train_start = 'none'
for input_dim, task_difficulty in tasks:
    output_dim = 4 if task_difficulty == 'hard' else 2
    input_constrs = int(np.log2(input_dim) + 2)
    output_constrs = int(np.log2(output_dim) + 2)
    data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
    property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
    h = EmbeddedNetwork(architecture=h_architectures[(input_dim, output_dim, depth)], name='h')

    for input_seed, output_seed in properties[(input_dim, task_difficulty)]:
        P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)
        
        print('=========================================================================================================================================================================================')
        print(f'unsafe -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
        print('=========================================================================================================================================================================================')
        log_file = f'{res_dir}/unsafe_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
        if not os.path.isfile(log_file):
            x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty, P=P, p=p, R=R, r=r)
            x_train_safe, y_train_safe = data_generator.enforce_safety(x_train=x_train, y_train=y_train, P=P, p=p, R=R, r=r)

            model_generator = ModelGenerator(x_train=x_train_safe, y_train=y_train_safe, 
                                             x_test=x_test, y_test=y_test, 
                                             x_test_inter=x_test_inter, y_test_inter=y_test_inter, 
                                             x_test_extra=x_test_extra, y_test_extra=y_test_extra,
                                             batch_size=batch_size, epochs=epochs, validation_split=validation_split, optimizer=optimizer, loss=loss, callbacks=[early_stopping],
                                             mode='regression')

            try: 
                unsafe = model_generator.train(log_file=log_file, model_type='unsafe', P=P, p=p, R=R, r=r, h=h)
                model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)
            except: 
                pass

##########################################
# POSTPROCESS & ORACLE
##########################################
depth = 3
box = 'none'
data_safety = 'unsafe'
train_start = 'none'

for input_dim, task_difficulty in tasks:
    output_dim = 4 if task_difficulty == 'hard' else 2
    input_constrs = int(np.log2(input_dim) + 2)
    output_constrs = int(np.log2(output_dim) + 2)
    data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
    property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
    h = EmbeddedNetwork(architecture=h_architectures[(input_dim, output_dim, depth)], name='h')

    input_seed, output_seed = 'none', 'none'
    P, p, R, r = 'none', 'none', 'none', 'none'

    x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty, P=P, p=p, R=R, r=r)

    model_generator = ModelGenerator(x_train=x_train, y_train=y_train, 
                                     x_test=x_test, y_test=y_test, 
                                     x_test_inter=x_test_inter, y_test_inter=y_test_inter, 
                                     x_test_extra=x_test_extra, y_test_extra=y_test_extra,
                                     batch_size=batch_size, epochs=epochs, validation_split=validation_split, optimizer=optimizer, loss=loss, callbacks=[early_stopping],
                                     mode='regression')

    print('=========================================================================================================================================================================================')
    print(f'unsafe -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
    print('=========================================================================================================================================================================================')
    log_file = f'{res_dir}/unsafe_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
    try: 
        unsafe = model_generator.train(log_file=None, model_type='unsafe', h=h)
        model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)
    except: 
        pass

    for input_seed, output_seed in properties[(input_dim, task_difficulty)]:
        P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)
        
        print('=========================================================================================================================================================================================')
        print(f'itp -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
        print('=========================================================================================================================================================================================')
        log_file = f'{res_dir}/itp_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
        if not os.path.isfile(log_file):
            try:
                itp = model_generator.train(log_file=log_file, model_type='itp', P=P, p=p, R=R, r=r, unsafe=unsafe)
                model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)
            except: 
                pass

        print('=========================================================================================================================================================================================')
        print(f'oracle -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
        print('=========================================================================================================================================================================================')
        log_file = f'{res_dir}/oracle_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
        if not os.path.isfile(log_file):
            try:
                oracle = model_generator.train(log_file=log_file, model_type='oracle', P=P, p=p, R=R, r=r)
                model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)
            except: 
                pass


##########################################
# SMLE
##########################################
#######################################################################################################
# Embeddeding Complexity
#######################################################################################################
depths = [1, 3, 5]
box = 'linear'
data_safety = 'unsafe'
train_start = 'none'

for depth in depths:
    for input_dim, task_difficulty in tasks:
        output_dim = 4 if task_difficulty == 'hard' else 2
        input_constrs = int(np.log2(input_dim) + 2)
        output_constrs = int(np.log2(output_dim) + 2)
        data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
        property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
        h = EmbeddedNetwork(architecture=h_architectures[(input_dim, output_dim, depth)], name='h')
        h_lower = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_lower', output_init=-1.)
        h_upper = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_upper', output_init=1.)
        
        input_seed, output_seed = 'none', 'none'
        P, p, R, r = 'none', 'none', 'none', 'none'

        x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty, P=P, p=p, R=R, r=r)
        
        model_generator = ModelGenerator(x_train=x_train, y_train=y_train, 
                                         x_test=x_test, y_test=y_test, 
                                         x_test_inter=x_test_inter, y_test_inter=y_test_inter, 
                                         x_test_extra=x_test_extra, y_test_extra=y_test_extra,
                                         batch_size=batch_size, epochs=epochs, validation_split=validation_split, optimizer=optimizer, loss=loss, callbacks=[early_stopping],
                                         mode='regression')



        for input_seed, output_seed in properties[(input_dim, task_difficulty)]:
            P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)

            print('=========================================================================================================================================================================================')
            print(f'smle -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
            print('=========================================================================================================================================================================================')
            log_file = f'{res_dir}/smle_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
            if not os.path.isfile(log_file):
                try: 
                    smle = model_generator.train(log_file=log_file, model_type='smle', P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper)
                    model_generator.test(log=log_file, P=P, p=p, R=R, r=r)
                except: 
                    pass


#######################################################################################################
# Auxiliary Complexity
#######################################################################################################
depth =  3
boxes = ['constant']
data_safety = 'unsafe'
train_start = 'none'

for box in boxes:
    for input_dim, task_difficulty in tasks:
        output_dim = 4 if task_difficulty == 'hard' else 2
        input_constrs = int(np.log2(input_dim) + 2)
        output_constrs = int(np.log2(output_dim) + 2)
        data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
        property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
        h = EmbeddedNetwork(architecture=h_architectures[(input_dim, output_dim, depth)], name='h')
        h_lower = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_lower', output_init=-1.)
        h_upper = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_upper', output_init=1.)

        input_seed, output_seed = 'none', 'none'
        P, p, R, r = 'none', 'none', 'none', 'none'

        x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty, P=P, p=p, R=R, r=r)
        
        model_generator = ModelGenerator(x_train=x_train, y_train=y_train, 
                                         x_test=x_test, y_test=y_test, 
                                         x_test_inter=x_test_inter, y_test_inter=y_test_inter, 
                                         x_test_extra=x_test_extra, y_test_extra=y_test_extra,
                                         batch_size=batch_size, epochs=epochs, validation_split=validation_split, optimizer=optimizer, loss=loss, callbacks=[early_stopping],
                                         mode='regression')


        for input_seed, output_seed in properties[(input_dim, task_difficulty)]:
            P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)

            print('=========================================================================================================================================================================================')
            print(f'smle -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
            print('=========================================================================================================================================================================================')
            log_file = f'{res_dir}/smle_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
            if not os.path.isfile(log_file):
                try: 
                    smle = model_generator.train(log_file=log_file, model_type='smle', P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper)
                    model_generator.test(log=log_file, P=P, p=p, R=R, r=r)
                except: 
                    pass


#######################################################################################################
# Data Safety
#######################################################################################################
depth =  3
box = 'linear'
data_safety = 'safe'
train_start = 'none'

for input_dim, task_difficulty in tasks:
    output_dim = 4 if task_difficulty == 'hard' else 2
    input_constrs = int(np.log2(input_dim) + 2)
    output_constrs = int(np.log2(output_dim) + 2)
    data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
    property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
    h = EmbeddedNetwork(architecture=h_architectures[(input_dim, output_dim, depth)], name='h')
    h_lower = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_lower', output_init=-1.)
    h_upper = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_upper', output_init=1.)

    for input_seed, output_seed in properties[(input_dim, task_difficulty)]:
        P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)

        print('=========================================================================================================================================================================================')
        print(f'smle -- input_dim: {input_dim} -- task_difficulty: {task_difficulty} -- input_seed: {input_seed} -- output_seed: {output_seed} --  depth: {depth} -- box: {box} -- data_safety: {data_safety} -- train_start: {train_start}')
        print('=========================================================================================================================================================================================')
        log_file = f'{res_dir}/smle_{input_dim}_{task_difficulty}_{input_seed}_{output_seed}_{depth}_{box}_{data_safety}_{train_start}.pkl'
        if not os.path.isfile(log_file):
            x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty, P=P, p=p, R=R, r=r)
            x_train_safe, y_train_safe = data_generator.enforce_safety(x_train=x_train, y_train=y_train, P=P, p=p, R=R, r=r)
        
            model_generator = ModelGenerator(x_train=x_train_safe, y_train=y_train_safe, 
                                             x_test=x_test, y_test=y_test, 
                                             x_test_inter=x_test_inter, y_test_inter=y_test_inter, 
                                             x_test_extra=x_test_extra, y_test_extra=y_test_extra,
                                             batch_size=batch_size, epochs=epochs, validation_split=validation_split, optimizer=optimizer, loss=loss, callbacks=[early_stopping],
                                             mode='regression')

            try: 
                smle = model_generator.train(log_file=log_file, model_type='smle', P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper)
                model_generator.test(log=log_file, P=P, p=p, R=R, r=r)
            except: 
                pass


