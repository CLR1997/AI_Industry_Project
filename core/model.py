import os
import sys
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
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
class SMLE(keras.Model):
    def __init__(self, mode, P, p, R, r, h, h_lower, h_upper, g=None, g_poly=None, safe_train=True, log_dir=None, **kwargs):
        super().__init__(**kwargs)
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

    
    def train_step(self, data):
        # Implement the Robust Training Algorithm (Algorithm 2)
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

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
