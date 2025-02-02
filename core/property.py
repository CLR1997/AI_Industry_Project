import pyomo.environ as pyo
import numpy as np 
from core.metrics import *
from core.model import Oracle

class Property:
    def __init__(self, P=None, p=None, R=None, r=None):
        self.P = P
        self.p = p
        self.R = R
        self.r = r

    def degeneracy(self):
        self.model = pyo.ConcreteModel()

        # Variables
        self.model.z = pyo.Var(range(self.R.shape[0]))

        # Constraints
        @self.model.Constraint(range(self.R.shape[1]))
        def out_poly(model, j):
            return model.z @ self.R[:,j] <= self.r[j]
        
        self.model.objective = pyo.Objective(expr=0, sense=pyo.minimize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        if  results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            return True
        else:
            return False

    def print(self, poly_type='input'):
        if poly_type == 'input':
            if self.P.shape[0] == 1:
                for j in range(self.P.shape[1]):
                   print(f'\n{self.P[0,j]}x <= {self.p[j]}')
            elif self.P.shape[0] == 2:
                for j in range(self.P.shape[1]):
                   print(f'\n{self.P[0,j]}x + {self.P[1,j]}y <= {self.p[j]}')
            else:
                for j in range(self.P.shape[1]):
                    exp = ''
                    for i in range(self.P.shape[0]):
                        exp = exp + f' {self.P[i,j]}x_{i} +'
                    exp = exp[:-2]
                    print(f'\n{exp} <= {self.p[j]}')

        elif poly_type == 'output':
            if self.R.shape[0] == 1:
                for j in range(self.R.shape[1]):
                   print(f'\n{self.R[0,j]}x <= {self.r[j]}')
            elif self.R.shape[0] == 2:
                for j in range(self.R.shape[1]):
                   print(f'\n{self.R[0,j]}x + {self.R[1,j]}y <= {self.r[j]}')
            else:
                for j in range(self.R.shape[1]):
                    exp = ''
                    for i in range(self.R.shape[0]):
                        exp = exp + f' {self.R[i,j]}x_{i} +'
                    exp = exp[:-2]
                    print(f'\n{exp} <= {self.r[j]}')


class Synthetic_Property(Property):
    def __init__(self, input_dim, input_constrs, output_dim, output_constrs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_constrs = input_constrs
        self.output_constrs = output_constrs

    def generate(self, input_seed, output_seed, input_bound=2., output_bound=None):
        # Input
        np.random.seed(input_seed)
        self.P = np.round(np.random.uniform(low=-1, high=1, size=(self.input_dim, self.input_constrs)), 2)
        self.p = np.round(np.random.uniform(low=-1, high=1, size=(self.input_constrs,)), 2)
        if input_bound:
            lhs_bounds = np.concatenate((np.eye(self.input_dim), -np.eye(self.input_dim)), axis=1)
            rhs_bounds = np.array([input_bound]*(2*self.input_dim))
            self.P = np.concatenate((self.P, lhs_bounds), axis=1)
            self.p = np.concatenate((self.p, rhs_bounds))

        # Output
        np.random.seed(output_seed)
        self.R = np.round(np.random.uniform(low=-1, high=1, size=(self.output_dim, self.output_constrs)), 2)
        self.r = np.round(np.random.uniform(low=-1, high=1, size=(self.output_constrs,)), 2)
        if output_bound:
            lhs_bounds = np.concatenate((np.eye(self.output_dim), -np.eye(self.output_dim)), axis=1)
            rhs_bounds = np.array([output_bound]*(2*self.output_dim))
            self.R = np.concatenate((self.R, lhs_bounds), axis=1)
            self.r = np.concatenate((self.r, rhs_bounds))

        return self.P, self.p, self.R, self.r


class RegRealistic_Property(Property):
    def __init__(self, wind_in, wind_out):
        super().__init__()
        self.wind_in = wind_in
        self.wind_out = wind_out

    def generate(self, series, q):
        magnitude = series.abs().max()
        deltas = np.abs(series[1:].values - series[:-1].values)
        max_delta = np.percentile(deltas, q=q)

        # Input
        self.P = np.concatenate((np.eye(self.wind_in), -np.eye(self.wind_in)), axis=1)
        self.p = np.concatenate(([magnitude]*self.wind_in, [magnitude]*self.wind_in))

        # Output
        self.R = np.concatenate(((np.eye(self.wind_out-1, self.wind_out, k=0) - np.eye(self.wind_out-1, self.wind_out, k=1)).T, 
                                 -(np.eye(self.wind_out-1, self.wind_out, k=0) - np.eye(self.wind_out-1, self.wind_out, k=1)).T), axis=1)
        self.r = np.concatenate(([max_delta] * (self.wind_out - 1), [max_delta] * (self.wind_out - 1)))

        return self.P, self.p, self.R, self.r

    def apply_differencing(self, x_train_delta, x_test_delta):
        P_delta = self.P
        p_delta = np.array([max(np.max(np.abs(x_train_delta)), np.max(np.abs(x_test_delta)))]*len(self.p))
        R_delta = self.R
        r_delta = self.r

        return P_delta, p_delta, R_delta, r_delta


class ClassRealistic_Property(Property):
    def __init__(self):
        super().__init__()

    def get_forbidden_pairs(self, y, q):
        counts = {}
        for c0 in range(y.shape[1]):
            for c1 in range(c0+1, y.shape[1]):
                counts[c0, c1] = np.sum(y[:, c0] + y[:, c1] == 2)
        thr = np.quantile(list(counts.values()), q=q)
        forbidden = set((c0, c1) for (c0, c1), v in counts.items() if v <= thr)

        return forbidden

    def generate(self, x, y, q):
        magnitude = np.max(np.abs(x))
        forbidden = self.get_forbidden_pairs(y=y, q=q)

        # Input
        self.P = np.concatenate((np.eye(x.shape[1]), -np.eye(x.shape[1])), axis=1)
        self.p = np.concatenate(([magnitude]*x.shape[1], [magnitude]*x.shape[1]))

        # Output
        self.R = np.zeros((y.shape[1], len(forbidden))) 
        for i, (c0, c1) in enumerate(forbidden):
            self.R[c0, i], self.R[c1, i] = 1., 1.
        self.r = np.ones(len(forbidden))

        return self.P, self.p, self.R, self.r
