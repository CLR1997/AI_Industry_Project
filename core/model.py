import os
import sys
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np

project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Add paths to Python path
sys.path.insert(0, os.path.join(project_root, 'core'))
sys.path.insert(0, project_root)

from core.optimization import *
np.seterr(invalid='ignore')


####################################################################################################
# Embeddings
####################################################################################################
class EmbeddedNetwork(keras.Model):
    def __init__(self, architecture, output_init=None, **kwargs):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.output_init = tf.keras.initializers.Constant(output_init) if output_init is not None else None
        self._layers = []

        if isinstance(self.architecture, list):
            for i, (units, layer_type) in enumerate(self.architecture[:-1]):
                if layer_type == 'lstm':
                    previous_layer_type, next_layer_type = self.architecture[i-1][1], self.architecture[i+1][1]
                    if previous_layer_type != 'lstm' or i == 0:
                        self._layers.append(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
                    if next_layer_type == 'lstm':
                        self._layers.append(keras.layers.LSTM(units=units, return_sequences=True))
                    else:
                        self._layers.append(keras.layers.LSTM(units=units, return_sequences=False))
                else:
                    self._layers.append(keras.layers.Dense(units=units, activation=layer_type))

            units, layer_type = self.architecture[-1]
            if self.output_init:
                self._layers.append(keras.layers.Dense(units=units, activation=layer_type, kernel_initializer='zeros', bias_initializer=self.output_init))
            else:
                if layer_type == 'lstm':
                    previous_layer_type = self.architecture[-2][1]
                    if previous_layer_type != 'lstm':
                        self._layers.append(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
                        self._layers.append(keras.layers.LSTM(units=units, return_sequences=False))
                    else:
                        self._layers.append(keras.layers.LSTM(units=units, return_sequences=False))
                else:
                    self._layers.append(keras.layers.Dense(units=units, activation=layer_type))

        elif isinstance(architecture, int):
            units, layer_type = self.architecture, 'linear'
            self._layers = [keras.layers.Dense(units=units, activation=layer_type, kernel_initializer='zeros', bias_initializer=self.output_init, trainable=False)]

    def call(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs


####################################################################################################
# Unsafe: underlying model for potprocess approach
####################################################################################################
class Unsafe(keras.Model):
    def __init__(self, mode, h, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.h = h
        self.g = keras.layers.Dense(units=output_dim, activation='linear', name='g')

    def call(self, inputs):
        outputs = self.g(self.h(inputs))

        if self.mode == 'regression':
            return outputs
        elif self.mode == 'multilabel-classification':
            return tf.nn.sigmoid(outputs)


####################################################################################################
# Oracle
####################################################################################################
class Oracle(keras.Model):
    def __init__(self, mode, P, p, R, r, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.P = P
        self.p = p
        self.R = R
        self.r = r

        self.OP = OutputProjection(R=R, r=r, mode=self.mode) 

    def call(self, inputs, outputs_true):
        outputs = []

        # Iterate over the input-true_output pairs and, in case of infeasibility, apply the MAP operator to restore it
        for k in range(inputs.shape[0]):
            i = tf.expand_dims(inputs[k], 0)
            o = tf.expand_dims(outputs_true[k], 0)

            input_membership = tf.reduce_all(i @ self.P <= self.p, axis=1)
            output_membership = tf.reduce_all(o @ self.R <= self.r, axis=1)
            if input_membership and not output_membership:
                o = tf.expand_dims(self.OP.project(tf.reshape(o, shape=(-1)).numpy()), 0)
            
            o = tf.cast(o, tf.float32)
            outputs.append(o)
        outputs = tf.concat(outputs, axis=0)

        return outputs


####################################################################################################
# Inference-Time Projector
####################################################################################################
class ITP(keras.Model):
    def __init__(self, mode, unsafe, P, p, R, r, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.P = P
        self.p = p
        self.R = R
        self.r = r

        self.OP = OutputProjection(R=R, r=r, mode=self.mode) 

        self.unsafe = unsafe
        self.unsafe.trainable = False

    def call(self, inputs):
        outputs = []

        # Iterate over the input-predicted_output pairs and, in case of infeasibility, apply the MAP operator to restore it
        for k in range(inputs.shape[0]):
            i = tf.expand_dims(inputs[k], 0)
            o = self.unsafe(i)
            
            input_membership = tf.reduce_all(i @ self.P <= self.p, axis=1)
            output_membership = tf.reduce_all(o @ self.R <= self.r, axis=1) if self.mode == 'regression' else tf.reduce_all(np.round(o) @ self.R <= self.r, axis=1)
            if input_membership and not output_membership:
                o = tf.expand_dims(self.OP.project(tf.reshape(o, shape=(-1)).numpy()), 0)
            
            o = tf.cast(o, tf.float32)
            outputs.append(o)
        outputs = tf.concat(outputs, axis=0)

        return outputs


####################################################################################################
# Safe
####################################################################################################
#### loss_type = 'mae' , 'mse' , 'lasso' , 'ridge' , 'elastic_net', 'l0_ridge'
#### dynamic = 'yes' , 'no'
class SMLE(keras.Model):
    def __init__(self, mode, P, p, R, r, h, h_lower, h_upper, loss_type = 'elastic_net', g=None, g_poly=None, safe_train=True, log_dir=None, 
                l0_pen=10.0, dynamic = 'no', lam=0.0001, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        #########################################################
        # For Colin: use your preferred variable names and values
        self.loss_type = loss_type
        self.l0_pen = l0_pen     #### penalization parameter for non zero parameter in the model 
        self.dynamic = dynamic
        if self.dynamic == 'no':
            self.lam = lam       #### lambda setting how strong the regularization is
            self.alpha = alpha   #### parameter determining the mixture between lasso and ridge regularization for the elastic net
        else:
            self.lam = tf.Variable(initial_value=lam, dtype=tf.float32, trainable=True, constraint=lambda x: tf.nn.relu(x))
            self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32, trainable=True, constraint=lambda x: tf.nn.relu(x))

        #########################################################

        self.mode = mode
        self.log_dir = log_dir
        self.P = P 
        self.p = p 
        self.R = R 
        self.r = r 
        self.violation = None
        self.safe_train = safe_train

        self.h = h
        self.h_lower = h_lower
        self.h_upper = h_upper 
        self.g = g if g is not None else keras.layers.Dense(self.R.shape[0], name='g')
        self.g_poly = g_poly if g_poly is not None else keras.layers.Dense(self.R.shape[0], name='g_poly')

        self.BP = BoundPropagator(P=P, p=p)
        self.CE = CounterExample(P=P, p=p, R=R, r=r, mode=self.mode)
        # Set the counterexample cash size to 1 or 10, as specified in the paragraph reporting the Q1 results
        self.CM = CashManager(max_size=1) if self.mode == 'regression' else CashManager(max_size=10)
        self.WP = WeightProjection(R=R, r=r, mode=self.mode)
        print(self.loss_type, self.lam, self.dynamic, self.alpha)
    #########################################################
    # For Colin: complete the following code snippet -- use your preferred function names and substitute "others" with all the additional parameters you may need
    def mae(self, y_true, y_pred):
        mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
        return mae
   
   #### defining the mse again is probably not needed, but this is to be thorough 
    def mse(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        #mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_pred))
        return mse 
    
    def ridge(self, y_true, y_pred, lam):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        ridge_loss = self.lam * tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables])
        return mse + ridge_loss 
    
    def ridge_dynamic(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        ridge_loss = self.lam * tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables])
        return mse + ridge_loss 
    
    def lasso(self, y_true, y_pred, lam):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        lasso_loss = self.lam * tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in self.trainable_variables]) 
        return mse + lasso_loss
    
    def lasso_dynamic(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        lasso_loss = self.lam * tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in self.trainable_variables]) 
        return mse + lasso_loss
    
    #### chosen here, unified elastic net. Can be expanded on by being fine tuned for alpha 
    def elastic_net(self, y_true, y_pred, lam, alpha):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        ridge_loss = self.lam * tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables])
        lasso_loss = self.lam * tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in self.trainable_variables]) 
        return mse + alpha * lasso_loss + (1-alpha) * ridge_loss
    
    def elastic_net_dynamic(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        ridge_loss = self.lam * tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables])
        lasso_loss = self.lam * tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in self.trainable_variables]) 
        return mse + self.alpha * lasso_loss + (1- self.alpha) * ridge_loss

    def l0_ridge(self, y_true, y_pred, lam, l0_pen):
        mse = tf.reduce_mean(tf.reduce_mean((y_true - y_pred) ** 2, axis=1))
        l0_loss = self.l0_pen * tf.reduce_sum([tf.reduce_sum(tf.cast(var != 0, tf.float32)) for var in self.trainable_variables])
        ridge_loss = self.lam * tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables])
        return mse + l0_loss + ridge_loss

    def train_step(self, data):
        # Implement the Robust Training Algorithm (Algorithm 2)
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x)
            #### y_pred = self(x)   to implement the regularization, I let the the input x be changed directly with the parameters.
            x_gt = self(x)
            #########################################################
            # For Colin: adjust the following code snippet as needed
            
            # if self.l0_r is not None and self.l0_r > 0:
            #      x_gt = tf.where(tf.abs(x_gt) < self.l0_r, tf.stop_gradient(x_gt * 0.0), x_gt)
            
            # y_pred = x_gt
            if self.dynamic == 'no':
                if self.loss_type == 'default' or self.loss_type == 'mse':
                    loss = self.mse(y_true=y, y_pred=y_pred)
                elif self.loss_type == 'mae':
                    loss = self.mae(y_true=y, y_pred=y_pred)
                elif self.loss_type == 'lasso':
                    loss = self.lasso(y_true=y, y_pred=x_gt, lam=self.lam)
                elif self.loss_type == 'ridge':
                    loss = self.ridge(y_true=y, y_pred=y_pred, lam = self.lam)
                elif self.loss_type == 'elastic_net':
                    loss = self.elastic_net(y_true=y, y_pred=y_pred, lam = self.lam, alpha = self.alpha)
                elif self.loss_type == 'l0_ridge':
                    loss = self.l0_ridge(y_true=y, y_pred=y_pred, lam = self.lam, l0_pen = self.l0_pen)
            elif self.dynamic == 'yes':
                if self.loss_type == 'lasso':
                    loss = self.lasso_dynamic(y_true=y, y_pred=x_gt)
                elif self.loss_type == 'ridge':
                    loss = self.ridge_dynamic(y_true=y, y_pred=y_pred)
                elif self.loss_type == 'elastic_net':
                    loss = self.elastic_net_dynamic(y_true=y, y_pred=y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        if self.dynamic == 'yes':
            print(f"Updated λ: {self.lam.numpy()}")

        if self.safe_train:
            W, w = self.g_poly.get_weights()
            if self.mode == 'regression':
                z_lower, z_upper = None, None
            if self.mode == 'multilabel-classification':
                z_lower, z_upper = self.BP.propagate(W=W, w=w, h_lower=self.h_lower, h_upper=self.h_upper)

            #################### Counter-Example Searching #########################
            y_counter, u = self.CE.generate(h_lower=self.h_lower, h_upper=self.h_upper, W=W, w=w, z_lower=z_lower)
            self.CM.push(y_counter)
            ########################################################################

            ######################## Weight Projection #############################
            self.violation = pyo.value(self.CE.model.objective)
            if self.violation > 0:
                W, w = self.WP.project(y=self.CM.pool, W=W, w=w, z_upper=z_upper)
                self.g_poly.set_weights([W, w])
            ########################################################################
     
        ############################# Logging ##################################
        sys.stdout.write(f'\r{90*" "}violation --> {self.violation}')
        sys.stdout.flush()
        ########################################################################
        
        ########################## Process Monitor #############################
        if self.log_dir:
            epoch = len(self.history.epoch)
            filename = f'{self.log_dir}/{epoch}.pkl'
            if not os.path.isfile(filename):
                log = {'gradients' : [g.numpy() for g in gradients], 'weights' : self.get_weights()}
                pickle.dump(log, open(filename, 'wb'))
        ########################################################################

        # Metric update
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


    def call(self, inputs):
        poly_membership = tf.expand_dims(tf.reduce_all(inputs @ self.P <= self.p, axis=1), axis=1)
        outputs = tf.where(poly_membership, 
                          self.g_poly(tf.maximum(tf.minimum(self.h(inputs), self.h_upper(inputs)), self.h_lower(inputs))), 
                          self.g(self.h(inputs)))

        if self.mode == 'regression':
            return outputs
        elif self.mode == 'multilabel-classification':
            return tf.nn.sigmoid(outputs)
 



