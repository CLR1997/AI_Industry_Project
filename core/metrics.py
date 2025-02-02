import numpy as np
from core.model import Oracle
from sklearn.metrics import accuracy_score, hamming_loss


def mse(y_true, y_pred):
    mse = np.mean(np.mean((y_true - y_pred)**2, axis=1))

    return mse


def r2(y_true, y_pred, y_var):
    res = np.mean((y_true - y_pred)**2, axis=0)
    r2 = np.mean(1 - res/y_var)

    return r2


def hamming(y_true, y_pred):
    hamming = hamming_loss(y_true=y_true, y_pred=y_pred)

    return hamming


def accuracy(y_true, y_pred):
    accuracy_byclass = []
    for c in range(y_true.shape[1]):
        accuracy_byclass.append(accuracy_score(y_true=y_true[:,c], y_pred=y_pred[:,c]))
    accuracy = np.mean(accuracy_byclass)

    return accuracy_byclass, accuracy


def property_difficulty(mode, P, p, R, r, x, y, y_var=None):
    oracle = Oracle(P=P, p=p, R=R, r=r, mode=mode)
    if mode == 'regression':
        difficulty = 1 - r2(y_true=y, y_pred=oracle(x, y), y_var=y_var) if len(x) != 0 else 1
    else:
        _, difficulty = accuracy(y_true=y, y_pred=oracle(x, y)) if len(x) != 0 else 1
        difficulty = 1 - difficulty

    return difficulty


def slack_tot_violation(P, p, R, r, x, y):
    input_membership = np.all(x @ P <= p, axis=1)
    violation = y @ R - r
    violation = violation[input_membership]
    violation = violation[violation > 10**(-6)]
    violation = np.sum(violation) if len(violation) > 0 else 0.0

    return violation


def slack_mean_violation(P, p, R, r, x, y, mode):
    input_membership = np.all(x @ P <= p, axis=1)
    violation = y @ R - r
    violation = violation[input_membership]
    if mode == 'regression':
        violation = violation[violation > 10**(-6)]
        violation = np.mean(violation) if len(violation) > 0 else 0.0
    else:
        violation = np.mean(np.sum(violation > 0, axis=1)/violation.shape[1])

    return violation
 

def membership_violation(P, p, R, r, x, y):
    input_membership = np.all(x @ P <= p, axis=1)
    output_membership = np.all(y @ R <= r + 10**(-6), axis=1)
    violation = np.sum((input_membership) & (~output_membership))/x.shape[0]

    return violation
