from datetime import datetime
import os
import pickle
import numpy as np

import sys

# Get absolute path to project root
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Add paths to Python path
sys.path.insert(0, os.path.join(project_root, 'core'))
sys.path.insert(0, project_root)



from core.data import *
from core.model import *
from core.property import *
from core.generate import *


#######################################################################################################
# Experiment Hyperparameters
#######################################################################################################
# FOR COLIN: The following are customizable parameters
t = 8  # an integer from 0 to 9
box = 'non-linear' # 'linear' or 'non-linear'
depth = 5 # 1, 3 or 5
input_seed = 0 # any integer <-- SKIP THIS FOR NOW
output_seed = 8 # any integer <-- SKIP THIS FOR NOW


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
# Task
##########################################
tasks = [(2,'easy'), (2, 'medium'), (2,'hard'), (4,'easy'), (4, 'medium'), (4,'hard'), (8,'easy'), (8, 'medium'), (8,'hard')]
input_dim, task_difficulty = tasks[t]
output_dim = 4 if task_difficulty == 'hard' else 2
data_generator = Synthetic_Data(train_size=train_size, test_size=test_size, input_dim=input_dim)
x_train, y_train, x_test, y_test, x_test_inter, y_test_inter, x_test_extra, y_test_extra = data_generator.generate(seed=data_seed, data_split=0.1, task_difficulty=task_difficulty)


##########################################
# Property
##########################################
input_constrs = int(np.log2(input_dim) + 2)
output_constrs = int(np.log2(output_dim) + 2)
property_generator = Synthetic_Property(input_dim=input_dim, input_constrs=input_constrs, output_dim=output_dim, output_constrs=output_constrs)
P, p, R, r = property_generator.generate(input_seed=input_seed, output_seed=output_seed)


##########################################
# Architectures
##########################################
h_architectures = {(input_dim, output_dim, depth) : [(width*input_dim*output_dim, 'relu') for width in range(1, int(depth/2)+1)] + [(width*input_dim*output_dim, 'relu') for width in range(int(depth/2)+1, 0, -1)] for input_dim in [2,4,8] for output_dim in [2,4] for depth in [1,3,5]}
h_aux_architectures = {
        **{(input_dim, output_dim, 'constant') : input_dim*output_dim for input_dim in [2,4,8] for output_dim in [2,4]},
        **{(input_dim, output_dim, 'linear') : [(input_dim*output_dim, 'linear')] for input_dim in [2,4,8] for output_dim in [2,4]},
        **{(input_dim, output_dim, 'non-linear') : [(input_dim*output_dim, 'linear'), (int(np.log2(input_dim*output_dim)), 'relu'), (input_dim*output_dim, 'linear')] for input_dim in [2,4,8] for output_dim in [2,4]}
        }


##########################################
# Training
##########################################
model_generator = ModelGenerator(x_train=x_train, y_train=y_train, 
                                 x_test=x_test, y_test=y_test, 
                                 x_test_inter=x_test_inter, y_test_inter=y_test_inter, 
                                 x_test_extra=x_test_extra, y_test_extra=y_test_extra,
                                 batch_size=batch_size, epochs=epochs, validation_split=validation_split, optimizer=optimizer, loss=loss, callbacks=[early_stopping],
                                 mode='regression')

h = EmbeddedNetwork(architecture=h_architectures[(input_dim, output_dim, depth)], name='h')
h_lower = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_lower', output_init=-1.)
h_upper = EmbeddedNetwork(architecture=h_aux_architectures[(input_dim, output_dim, box)], name='h_upper', output_init=1.)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
log_file = f"{timestamp}.pkl"

model = model_generator.train(log_file=log_file, model_type='smle',P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper, safe_train=False)
#log = model_generator.test(log_file=log_file, P=P, p=p, R=R, r=r)
log = model_generator.test(log_file=log_file)

#vector_lam = tf.constant([0.01, 0.1, 0.5, 1.0], dtype=tf.float32)
#for i in range(4):
#        lam_gt = vector_lam[i]
#        model = model_generator.train(log_file=log_file, model_type='smle',P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper, 
#                                      loss_type = 'lasso', lam = lam_gt, safe_train=False)
#        log = model_generator.test(log_file=log_file)

