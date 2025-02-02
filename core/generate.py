from core.model import *
from core.metrics import *
from core.data import *
from sklearn.metrics import mean_squared_error, r2_score
import pyomo.environ as pyo
import pickle
import time


class ModelGenerator:
    def __init__(self, mode, x_train, y_train, x_test, y_test, x_test_inter=None, y_test_inter=None, x_test_extra=None, y_test_extra=None, 
                 batch_size=None, epochs=None, validation_split=None, optimizer=None, loss=None, metrics=None, callbacks=None):
        self.mode = mode
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_test_inter = x_test_inter
        self.y_test_inter = y_test_inter
        self.x_test_extra = x_test_extra
        self.y_test_extra = y_test_extra
        self.input_dim = x_train.shape[1]
        self.output_dim = y_train.shape[1]
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.optimizer = optimizer
        self.loss = loss 
        self.metrics = metrics
        self.callbacks = callbacks 


    def train(self, model_type, P=None, p=None, R=None, r=None, h=None, h_lower=None, h_upper=None, g=None, g_poly=None, unsafe=None, log_file=None, safe_train=True): 
        start_time = time.time()
        
        if model_type == 'oracle':
            model = Oracle(mode=self.mode, P=P, p=p, R=R, r=r)

        elif model_type == 'unsafe':
            model = Unsafe(mode=self.mode, h=h, output_dim=self.output_dim)
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, run_eagerly=True)
            model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split, callbacks=self.callbacks)

        elif model_type == 'itp':
            model = ITP(mode=self.mode, unsafe=unsafe, P=P, p=p, R=R, r=r)

        elif model_type == 'smle':
            model = SMLE(mode=self.mode, P=P, p=p, R=R, r=r, h=h, h_lower=h_lower, h_upper=h_upper, g=g, g_poly=g_poly, safe_train=safe_train)
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, run_eagerly=True)
            model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split, callbacks=self.callbacks)

            i = None
            if safe_train:
                #################### Post-Training Safety Enforcing #########################
                W, w = model.g_poly.get_weights()
                if self.mode == 'regression':
                    z_lower, z_upper = None, None
                if self.mode == 'multilabel-classification':
                    z_lower, z_upper = model.BP.propagate(W=W, w=w, h_lower=model.h_lower, h_upper=model.h_upper)
                y_counter, u = model.CE.generate(h_lower=model.h_lower, h_upper=model.h_upper, W=W, w=w, z_lower=z_lower)
                model.violation = pyo.value(model.CE.model.objective)

                i = 0
                while model.violation > 10**-6:
                    model.CM.push(y_counter)
                    W, w = model.WP.project(y=model.CM.pool, W=W, w=w, z_upper=z_upper)
                    model.g_poly.set_weights([W, w])
                    if self.mode == 'regression':
                        z_lower, z_upper = None, None
                    elif self.mode == 'multilabel-classification':
                        z_lower, z_upper = model.BP.propagate(W=W, w=w, h_lower=model.h_lower, h_upper=model.h_upper)
                    y_counter, u = model.CE.generate(h_lower=model.h_lower, h_upper=model.h_upper, W=W, w=w, z_lower=z_lower)
                    model.violation = pyo.value(model.CE.model.objective)
                    print(f'violation --> {model.violation}')
                    i += 1
                print(f'\n# post-training projections: {i}')
                ########################################################################
        end_time = time.time()
                
        log = {}
        log['dimension'] = (self.input_dim, self.output_dim)
        log['P'] = P
        log['p'] = p
        log['R'] = R
        log['r'] = r
        log['loss'] = model.history.history['loss'] if model_type not in ['oracle', 'itp'] else None
        log['val_loss'] = model.history.history['val_loss'] if model_type not in ['oracle', 'itp'] else None
        log['projections'] = i if model_type == 'smle' else None
        log['theoretical_slack_violation'] =  model.violation if model_type == 'smle' else None
        log['final_weights'] = model.get_weights()
        log['train_time'] = end_time - start_time
        log['y_train'] = self.y_train
        log['y_train_pred'] = model(self.x_train) if model_type != 'oracle' else model(self.x_train, self.y_train)
        log['y_test'] = self.y_test
        log['y_test_pred'] = model(self.x_test) if model_type != 'oracle' else model(self.x_test, self.y_test)
        if self.y_test_inter is not None and len(self.y_test_inter) > 0:
            log['y_test_inter'] = self.y_test_inter
            log['y_test_inter_pred'] = model(self.x_test_inter) if model_type != 'oracle' else model(self.x_test_inter, self.y_test_inter)
        if self.y_test_extra is not None and len(self.y_test_extra) > 0:
            log['y_test_extra'] = self.y_test_extra
            log['y_test_extra_pred'] = model(self.x_test_extra) if model_type != 'oracle' else model(self.x_test_extra, self.y_test_extra)

        if self.mode == 'multilabel-classification':
            log['y_train_pred'] = np.round(log['y_train_pred']) 
            log['y_test_pred'] = np.round(log['y_test_pred'])
            if self.y_test_inter is not None and len(self.y_test_inter) > 0:
                log['y_test_inter_pred'] = np.round(log['y_test_inter_pred'])
            if self.y_test_extra is not None and len(self.y_test_extra) > 0:
                log['y_test_extra_pred'] = np.round(log['y_test_extra_pred'])
        
        if log_file:
            pickle.dump(log, open(log_file, 'wb'))
            
        return model

    def reverse_differencing(self, log_file, x_train, y_train, x_test, y_test):

        rrd = RegRealistic_Data()

        log = pickle.load(open(log_file, 'rb'))
        log['y_train_delta'] = log['y_train']
        log['y_train_delta_pred'] = log['y_train_pred']
        log['y_test_delta'] = log['y_test']
        log['y_test_delta_pred'] = log['y_test_pred']
        log['mse_test_delta'] = mse(y_true=log['y_test_delta'], y_pred=log['y_test_delta_pred'])
        log['r2_test_delta'] = r2(y_true=log['y_test_delta'], y_pred=log['y_test_delta_pred'], y_var=np.var(log['y_test_delta'], axis=0))
        log['y_train'] = y_train
        log['y_train_pred'] = rrd.reverse_differencing_y(x=x_train, y_delta=log['y_train_delta_pred'])
        log['y_test'] = y_test
        log['y_test_pred'] = rrd.reverse_differencing_y(x=x_test, y_delta=log['y_test_delta_pred'])
        pickle.dump(log, open(log_file, 'wb'))


    def test(self, log_file, P=None, p=None, R=None, r=None, print_log=True):

        log = pickle.load(open(log_file, 'rb'))
        
        if self.mode == 'regression':
            log['mse_test'] = mse(y_true=self.y_test, y_pred=log['y_test_pred'])
            log['r2_test'] = r2(y_true=self.y_test, y_pred=log['y_test_pred'], y_var=np.var(self.y_test, axis=0))
        else: 
            log['accuracy_test_byclass'], log['accuracy_test'] = accuracy(y_true=self.y_test, y_pred=log['y_test_pred'])
            log['hamming_test'] = hamming(y_true=self.y_test, y_pred=log['y_test_pred'])
        if P is not None:
            log['data_test_slack_tot_violation'] = slack_tot_violation(P, p, R, r, x=self.x_test, y=self.y_test)
            log['model_test_slack_tot_violation'] = slack_tot_violation(P, p, R, r, x=self.x_test, y=log['y_test_pred'])
            log['data_test_slack_mean_violation'] = slack_mean_violation(P, p, R, r, x=self.x_test, y=self.y_test, mode=self.mode)
            log['model_test_slack_mean_violation'] = slack_mean_violation(P, p, R, r, x=self.x_test, y=log['y_test_pred'], mode=self.mode)
            log['data_test_membership_violation'] = membership_violation(P, p, R, r, x=self.x_test, y=self.y_test)
            log['model_test_membership_violation'] = membership_violation(P, p, R, r, x=self.x_test, y=log['y_test_pred'])
        else:
            log['data_test_slack_tot_violation'] = None
            log['model_test_slack_tot_violation'] = None
            log['data_test_slack_mean_violation'] = None
            log['model_test_slack_mean_violation'] = None
            log['data_test_membership_violation'] = None
            log['model_test_membership_violation'] = None

        if self.y_test_inter is not None and len(self.y_test_inter):
            if self.mode == 'regression':
                log['mse_test_inter'] = mse(y_true=self.y_test_inter, y_pred=log['y_test_inter_pred'])
                log['r2_test_inter'] = r2(y_true=self.y_test_inter, y_pred=log['y_test_inter_pred'], y_var=np.var(self.y_test, axis=0))
            else:
                log['accuracy_test_inter_byclass'], log['accuracy_test_inter'] = accuracy(y_true=self.y_test_inter, y_pred=log['y_test_inter_pred'])
                log['hamming_test_inter'] = hamming(y_true=self.y_test_inter, y_pred=log['y_test_inter_pred'])
            if P is not None:
                log['data_test_inter_slack_tot_violation'] = slack_tot_violation(P, p, R, r, x=self.x_test_inter, y=self.y_test_inter)
                log['model_test_inter_slack_tot_violation'] = slack_tot_violation(P, p, R, r, x=self.x_test_inter, y=log['y_test_inter_pred'])
                log['data_test_inter_slack_mean_violation'] = slack_mean_violation(P, p, R, r, x=self.x_test_inter, y=self.y_test_inter, mode=self.mode)
                log['model_test_inter_slack_mean_violation'] = slack_mean_violation(P, p, R, r, x=self.x_test_inter, y=log['y_test_inter_pred'], mode=self.mode)
                log['data_test_inter_membership_violation'] = membership_violation(P, p, R, r, x=self.x_test_inter, y=self.y_test_inter)
                log['model_test_inter_membership_violation'] = membership_violation(P, p, R, r, x=self.x_test_inter, y=log['y_test_inter_pred'])
            else:
                log['data_test_inter_slack_tot_violation'] = None
                log['model_test_inter_slack_tot_violation'] = None
                log['data_test_inter_slack_mean_violation'] = None
                log['model_test_inter_slack_mean_violation'] = None
                log['data_test_inter_membership_violation'] = None
                log['model_test_inter_membership_violation'] = None
        else:
            log['mse_test_inter'] = None
            log['r2_test_inter'] = None
            log['accuracy_test_inter'] = None
            log['hamming_test_inter'] = None
            log['data_test_inter_slack_tot_violation'] = None
            log['model_test_inter_slack_tot_violation'] = None
            log['data_test_inter_slack_mean_violation'] = None
            log['model_test_inter_slack_mean_violation'] = None
            log['data_test_inter_membership_violation'] = None
            log['model_test_inter_membership_violation'] = None

        if self.y_test_extra is not None and len(self.y_test_extra):
            if self.mode == 'regression':
                log['mse_test_extra'] = mse(y_true=self.y_test_extra, y_pred=log['y_test_extra_pred'])
                log['r2_test_extra'] = r2(y_true=self.y_test_extra, y_pred=log['y_test_extra_pred'], y_var=np.var(self.y_test, axis=0))
            else:
                log['accuracy_test_extra_byclass'], log['accuracy_test_extra'] = accuracy(y_true=self.y_test_extra, y_pred=log['y_test_extra_pred'])
                log['hamming_test_extra'] = hamming(y_true=self.y_test_extra, y_pred=log['y_test_extra_pred'])
            if P is not None:
                log['data_test_extra_slack_tot_violation'] = slack_tot_violation(P, p, R, r, x=self.x_test_extra, y=self.y_test_extra)
                log['model_test_extra_slack_tot_violation'] = slack_tot_violation(P, p, R, r, x=self.x_test_extra, y=log['y_test_extra_pred'])
                log['data_test_extra_slack_mean_violation'] = slack_mean_violation(P, p, R, r, x=self.x_test_extra, y=self.y_test_extra, mode=self.mode)
                log['model_test_extra_slack_mean_violation'] = slack_mean_violation(P, p, R, r, x=self.x_test_extra, y=log['y_test_extra_pred'], mode=self.mode)
                log['data_test_extra_membership_violation'] = membership_violation(P, p, R, r, x=self.x_test_extra, y=self.y_test_extra)
                log['model_test_extra_membership_violation'] = membership_violation(P, p, R, r, x=self.x_test_extra, y=log['y_test_extra_pred'])
            else:
                log['data_test_extra_slack_tot_violation'] = None
                log['model_test_extra_slack_tot_violation'] = None
                log['data_test_extra_slack_mean_violation'] = None
                log['model_test_extra_slack_mean_violation'] = None
                log['data_test_extra_membership_violation'] = None
                log['model_test_extra_membership_violation'] = None
        else:
            log['mse_test_extra'] = None
            log['r2_test_extra'] = None
            log['accuracy_test_extra'] = None
            log['hamming_test_extra'] = None
            log['data_test_extra_slack_tot_violation'] = None
            log['model_test_extra_slack_tot_violation'] = None
            log['data_test_extra_slack_mean_violation'] = None
            log['model_test_extra_slack_mean_violation'] = None
            log['data_test_extra_membership_violation'] = None
            log['model_test_extra_membership_violation'] = None

        if print_log:
            print(f'train_time --> {log["train_time"]}')
            print(f'theoretical_slack_violation --> {log["theoretical_slack_violation"]}')
            print('\n================= IN DISTRIBUTION ====================')
            print(f'data_slack_tot_violation --> {log["data_test_slack_tot_violation"]}')
            print(f'model_slack_tot_violation --> {log["model_test_slack_tot_violation"]}')
            print(f'data_slack_mean_violation --> {log["data_test_slack_mean_violation"]}')
            print(f'model_slack_mean_violation --> {log["model_test_slack_mean_violation"]}')
            print(f'data_membership_violation --> {log["data_test_membership_violation"]}')
            print(f'model_membership_violation --> {log["model_test_membership_violation"]}')
            if self.mode == 'regression':
                print(f'MSE --> {log["mse_test"]}')
                print(f'R2 --> {log["r2_test"]}')
            else:
                print(f'accuracy --> {log["accuracy_test"]}')
                print(f'accuracy_byclass --> {log["accuracy_test_byclass"]}')
                print(f'hamming --> {log["hamming_test"]}')
            if self.y_test_inter is not None and len(self.y_test_inter) > 0:
                print('\n================= INTERPOLATION ====================')
                print(f'data_slack_tot_violation --> {log["data_test_inter_slack_tot_violation"]}')
                print(f'model_slack_tot_violation --> {log["model_test_inter_slack_tot_violation"]}')
                print(f'data_slack_mean_violation --> {log["data_test_inter_slack_mean_violation"]}')
                print(f'model_slack_mean_violation --> {log["model_test_inter_slack_mean_violation"]}')
                print(f'data_membership_violation --> {log["data_test_inter_membership_violation"]}')
                print(f'model_membership_violation --> {log["model_test_inter_membership_violation"]}')
                if self.mode == 'regression':
                    print(f'MSE --> {log["mse_test_inter"]}')
                    print(f'R2 --> {log["r2_test_inter"]}')
                else:
                    print(f'accuracy --> {log["accuracy_test_inter"]}')
                    print(f'accuracy_byclass --> {log["accuracy_test_inter_byclass"]}')
                    print(f'hamming --> {log["hamming_test_inter"]}')
            if self.y_test_extra is not None and len(self.y_test_extra) > 0:
                print('\n================= EXTRAPOLATION ====================')
                print(f'data_slack_tot_violation --> {log["data_test_extra_slack_tot_violation"]}')
                print(f'model_slack_tot_violation --> {log["model_test_extra_slack_tot_violation"]}')
                print(f'data_slack_mean_violation --> {log["data_test_extra_slack_mean_violation"]}')
                print(f'model_slack_mean_violation --> {log["model_test_extra_slack_mean_violation"]}')
                print(f'data_membership_violation --> {log["data_test_extra_membership_violation"]}')
                print(f'model_membership_violation --> {log["model_test_extra_membership_violation"]}')
                if self.mode == 'regression':
                    print(f'MSE --> {log["mse_test_extra"]}')
                    print(f'R2 --> {log["r2_test_extra"]}')
                else:
                    print(f'accuracy --> {log["accuracy_test_extra"]}')
                    print(f'accuracy_byclass --> {log["accuracy_test_extra_byclass"]}')
                    print(f'hamming --> {log["hamming_test_extra"]}')
                print('\n====================================================')

        return log
