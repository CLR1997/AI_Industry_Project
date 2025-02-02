import os
import pickle
import numpy as np
from core.data import *
from core.property import *
from core.metrics import *


#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
name = 'synthetic'
prep_dir = f'benchmarks/{name}/preprocess'

##########################################
# Data
##########################################
data_seed = 0
train_size = 5000
test_size = 1000
search_seeds = (150, 150)
 
##########################################
# Tasks
##########################################
tasks = [(2,'easy'), (2,'medium'), (2,'hard'), 
         (4,'easy'), (4,'medium'), (4,'hard'),
         (8,'easy'), (8,'medium'), (8,'hard')]


##########################################
# Properties 
##########################################
ispath = os.path.isfile(f'{prep_dir}/properties.pkl')
if ispath:
    properties = pickle.load(open(f'{prep_dir}/properties.pkl', 'rb'))
else:
    properties = {task : np.array([], dtype=[('seeds', tuple), ('difficulty', float), ('unsafety', float)]) for task in tasks}


for input_dim, task_difficulty in tasks:
    output_dim = 4 if task_difficulty == 'hard' else 2
    output_constrs = int(np.log2(output_dim) + 2)
    print('================================================================')
    print(f'input_dim : {input_dim} -- task_difficulty : {task_difficulty}')
    print('================================================================')
    input_constrs = int(np.log2(input_dim) + 2)
    property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
    data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
    _, _, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty)


    for input_seed in range(search_seeds[0]):
        for output_seed in range(search_seeds[1]):
            seeds = input_seed, output_seed
            print(seeds)
            if seeds not in list(properties[input_dim, task_difficulty]['seeds']):
                P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)
                if not property_generator.degeneracy():
                    test_violation = slack_tot_violation(P=P, p=p, R=R, r=r, x=x_test, y=y_test)
                    test_inter_violation = slack_tot_violation(P=P, p=p, R=R, r=r, x=x_test_inter, y=y_test_inter)
                    test_extra_violation = slack_tot_violation(P=P, p=p, R=R, r=r, x=x_test_extra, y=y_test_extra)
                    if test_violation * test_inter_violation * test_extra_violation != 0:
                        violation = test_violation + test_inter_violation + test_extra_violation
                        test_difficulty = property_difficulty(P=P, p=p, R=R, r=r, x=x_test, y=y_test, y_var=np.var(y_test, axis=0), mode='regression')
                        test_inter_difficulty = property_difficulty(P=P, p=p, R=R, r=r, x=x_test_inter, y=y_test_inter, y_var=np.var(y_test_inter, axis=0), mode='regression')
                        test_extra_difficulty = property_difficulty(P=P, p=p, R=R, r=r, x=x_test_extra, y=y_test_extra, y_var=np.var(y_test_extra, axis=0), mode='regression')
                        if (test_difficulty <= 1) & (test_inter_difficulty <= 1) & (test_extra_difficulty <= 1):
                            prop = []
                            difficulty = (test_difficulty + test_inter_difficulty + test_extra_difficulty)/3
                            print((seeds, difficulty, violation))
                            prop = [(seeds, difficulty, violation)]
                            prop = np.array(prop, dtype=[('seeds', tuple), ('difficulty', float), ('unsafety', float)])
                            properties[input_dim, task_difficulty] = np.concatenate((properties[input_dim, task_difficulty], prop))
                            pickle.dump(properties, open(f'{prep_dir}/properties.pkl', 'wb'))
