import pyomo.environ as pyo
from pyomo.common.config import ConfigValue, ConfigDict
import numpy as np
import omlt 
import omlt.io
import omlt.neuralnet
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


config = ConfigDict()
config.declare("bound tolerance", ConfigValue(
    default=1E-6,
    domain=float,
    description="Bound tolerance",
    doc="Relative tolerance for bound feasibility checks"
))


class CashManager:
    # Implement the caching mechanism, used in the Projection with Delayed Constraint Generation (Algorithm 1)
    def __init__(self, max_size):
        self.max_size = max_size
        self.pool = None

    def push(self, y_counter):
        if self.pool is None:
            self.pool = y_counter.reshape(1, -1)
        elif self.pool.shape[0] < self.max_size:
            self.pool = np.vstack([self.pool, y_counter])
        else:
            self.pool = np.vstack([self.pool, y_counter])
            self.pool = self.pool[1:]


class BoundPropagator:
    def __init__(self, P, p):
        self.P = P
        self.p = p
        
        # Blocks
        self.model = pyo.ConcreteModel()
        self.model.h_lower = omlt.OmltBlock()
        self.model.h_upper = omlt.OmltBlock()

    def propagate(self, W, w, h_lower, h_upper):
        # Variables
        self.model.x = pyo.Var(range(self.P.shape[0]))
        self.model.y = pyo.Var(range(W.shape[0]))
        self.model.y_lower = pyo.Var(range(W.shape[0]))
        self.model.y_upper = pyo.Var(range(W.shape[0]))
        self.model.t = pyo.Var(range(W.shape[0]), domain=pyo.Binary) # Auxiliary variable to handle box degeneracy
        self.model.z = pyo.Var(range(W.shape[1]))

        # Encode auxiliary networks
        formulation = omlt.neuralnet.ReluComplementarityFormulation(omlt.io.load_keras_sequential(h_lower))
        self.model.h_lower.build_formulation(formulation)

        @self.model.Constraint(range(self.P.shape[0]))
        def h_lower_input(model, j):
            return model.x[j] == model.h_lower.inputs[j]
        @self.model.Constraint(range(W.shape[0]))
        def h_lower_output(model, j):
            return model.y_lower[j] == model.h_lower.outputs[j]
        
        formulation = omlt.neuralnet.ReluComplementarityFormulation(omlt.io.load_keras_sequential(h_upper))
        self.model.h_upper.build_formulation(formulation)
        
        @self.model.Constraint(range(self.P.shape[0]))
        def h_upper_input(model, j):
            return model.x[j] == model.h_upper.inputs[j]
        @self.model.Constraint(range(W.shape[0]))
        def h_upper_output(model, j):
            return model.y_upper[j] == model.h_upper.outputs[j]

        # Constraints
        @self.model.Constraint(range(self.P.shape[1]))
        def in_poly(model, j):
            return model.x @ self.P[:,j] <= self.p[j]
        @self.model.Constraint(range(W.shape[0]))
        def lower_clip(model, j):
            return model.y[j] >= model.y_lower[j] 
        @self.model.Constraint(range(W.shape[0]))
        def upper_clip(model, j):
            return model.y[j] <= (1 - model.t[j]) * model.y_lower[j] + model.t[j] * model.y_upper[j] 
        @self.model.Constraint(range(W.shape[1]))
        def g(model, j):
            return model.z[j] == model.y @ W[:,j] + w[j]

        z_lower = []
        z_upper = []
        for j in range(W.shape[1]):
            self.model.objective = pyo.Objective(expr=self.model.z[j], sense=pyo.minimize)
            solver = pyo.SolverFactory('gurobi')
            results = solver.solve(self.model)
            z_lower += [self.model.z[j].value]
            
            self.model.objective = pyo.Objective(expr=self.model.z[j], sense=pyo.maximize)
            solver = pyo.SolverFactory('gurobi')
            results = solver.solve(self.model)
            z_upper += [self.model.z[j].value]

        z_lower, z_upper = np.array(z_lower), np.array(z_upper)
        
        return z_lower, z_upper
    
    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})


class CounterExample:
    # Implement the counterexample generator used in the Projection with Delayed Constraint Generation (Algorithm 1)
    def __init__(self, P, p, R, r, mode):
        self.mode = mode
        self.P = P
        self.p = p
        self.R = R
        self.r = r

    def generate(self, h_lower, h_upper, W, w, z_lower=None):
        # Blocks
        self.model = pyo.ConcreteModel()
        self.model.h_lower = omlt.OmltBlock()
        self.model.h_upper = omlt.OmltBlock()

        # Variables
        self.model.x = pyo.Var(range(self.P.shape[0]))
        self.model.y = pyo.Var(range(W.shape[0]))
        self.model.y_lower = pyo.Var(range(W.shape[0]))
        self.model.y_upper = pyo.Var(range(W.shape[0]))
        self.model.t = pyo.Var(range(W.shape[0]), domain=pyo.Binary) # Auxiliary variable to handle box degeneracy
        self.model.z = pyo.Var(range(W.shape[1]))
        if self.mode == 'regression':
            self.model.u = pyo.Var(range(self.R.shape[1])) # Continuos variables to represent the violations
            self.model.b = pyo.Var(range(self.R.shape[1]), domain=pyo.Binary) # Auxiliary variables to ensure proper violations
        elif self.mode == 'multilabel-classification':
            self.model.u = pyo.Var(range(self.R.shape[1]), domain=pyo.Binary) # Binary variables to represent the violations
            self.model.d = pyo.Var(range(self.R.shape[1])) # Continuous variables to represent translations
            self.model.b = pyo.Var(range(self.R.shape[0]), domain=pyo.Binary) # Auxiliary variables to ensure proper violations 

        # Encode auxiliary networks
        formulation = omlt.neuralnet.ReluComplementarityFormulation(omlt.io.load_keras_sequential(h_lower))
        self.model.h_lower.build_formulation(formulation)

        @self.model.Constraint(range(self.P.shape[0]))
        def h_lower_input(model, j):
            return model.x[j] == model.h_lower.inputs[j]
        @self.model.Constraint(range(W.shape[0]))
        def h_lower_output(model, j):
            return model.y_lower[j] == model.h_lower.outputs[j]
        
        formulation = omlt.neuralnet.ReluComplementarityFormulation(omlt.io.load_keras_sequential(h_upper))
        self.model.h_upper.build_formulation(formulation)
        
        @self.model.Constraint(range(self.P.shape[0]))
        def h_upper_input(model, j):
            return model.x[j] == model.h_upper.inputs[j]
        @self.model.Constraint(range(W.shape[0]))
        def h_upper_output(model, j):
            return model.y_upper[j] == model.h_upper.outputs[j]

        # Constraints
        @self.model.Constraint(range(self.P.shape[1]))
        def in_poly(model, j):
            return model.x @ self.P[:,j] <= self.p[j]
        @self.model.Constraint(range(W.shape[0]))
        def lower_clip(model, j):
            return model.y[j] >= model.y_lower[j] 
        @self.model.Constraint(range(W.shape[0]))
        def upper_clip(model, j):
            return model.y[j] <= (1 - model.t[j]) * model.y_lower[j] + model.t[j] * model.y_upper[j] 
        @self.model.Constraint(range(W.shape[1]))
        def g(model, j):
            return model.z[j] == model.y @ W[:,j] + w[j]
        if self.mode == 'regression':
            @self.model.Constraint(range(self.R.shape[1]))
            def violation(model, j):
                return model.u[j] == model.z @ self.R[:,j] - self.r[j]
        elif self.mode == 'multilabel-classification':
            @self.model.Constraint(range(self.R.shape[0]))
            def violation_a(model, j): 
                return model.z[j] >= -model.b[j] * (z_lower[j] - 10**-5) + z_lower[j]
            @self.model.Constraint(range(self.R.shape[1]))
            def violation_b(model, j): 
                return model.u[j] <= 1/2 * (model.b @ self.R[:,j])
#           self.model.violation_c = pyo.Constraint(expr=sum(self.model.u[j] for j in range(self.R.shape[1])) <= 1)
            @self.model.Constraint(range(self.R.shape[1]), range(self.R.shape[0]))
            def violation_d(model, j, i): 
                if self.R[i,j] != 0:
                    return model.d[j] <= model.z[i]
                else: 
                    return pyo.Constraint.Feasible

        # Objective
        if self.mode == 'regression':
            expr = sum(self.model.u[j] * self.model.b[j] for j in range(self.R.shape[1]))
        elif self.mode == 'multilabel-classification':
            expr = sum(self.model.u[j] * self.model.d[j] for j in range(self.R.shape[1]))
        self.model.objective = pyo.Objective(expr=expr, sense=pyo.maximize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        y = np.array([self.model.y[j].value for j in range(W.shape[0])])
        u = np.array([self.model.u[j].value for j in range(self.R.shape[1])])

        return y, u

    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})
    

class WeightProjection:
    # Implement the weight projector (eq. 12), used in the Projection with Delayed Constraint Generation (Algorithm 1)
    def __init__(self, R, r, mode):
        self.mode = mode
        self.R = R
        self.r = r

    def project(self, y, W, w, z_upper=None):
        self.model = pyo.ConcreteModel()

        # Variables
        self.model.W = pyo.Var(range(W.shape[0]), range(W.shape[1]))
        self.model.w = pyo.Var(range(W.shape[1]))
        self.model.z = pyo.Var(range(y.shape[0]), range(W.shape[1]))
        if self.mode == 'multilabel-classification':
            self.model.b = pyo.Var(range(y.shape[0]), range(W.shape[1]), domain=pyo.Binary)

        # Constraints
        @self.model.Constraint(range(y.shape[0]), range(W.shape[1]))
        def g(model, i, j):
            return model.z[i,j] == sum(y[i,k] * model.W[k,j] for k in range(W.shape[0])) + model.w[j] 
        if self.mode == 'multilabel-classification':
            @self.model.Constraint(range(y.shape[0]), range(W.shape[1]))
            def forbidden_pairs(model, i, j): 
                if z_upper[j] <= 0:
                    return model.z[i,j] <= z_upper[j]
                else:
                    return model.z[i,j] <= model.b[i,j] * z_upper[j] - (10**-5 * z_upper[j])
        @self.model.Constraint(range(y.shape[0]), range(self.R.shape[1]))
        def property(model, i, j):
            if self.mode == 'regression':
                return sum(model.z[i,k] * self.R[k,j] for k in range(self.R.shape[0])) <= self.r[j]
            elif self.mode == 'multilabel-classification':
                return sum(model.b[i,k] * self.R[k,j] for k in range(self.R.shape[0])) <= self.r[j]

        # Objective
        expr = sum(sum((W[i,j] - self.model.W[i,j])**2 for i in range(W.shape[0])) for j in range(W.shape[1]))
        expr += sum((w[i] - self.model.w[i])**2 for i in range(w.shape[0]))
        self.model.objective = pyo.Objective(expr=expr, sense=pyo.minimize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        W = np.array([[self.model.W[i,j].value for j in range(W.shape[1])] for i in range(W.shape[0])])
        w = np.array([self.model.w[j].value for j in range(w.shape[0])])

        return W, w

    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})


class OutputProjection:
    # Implement the MAP operator described in the Experimentation Section (eq 23)
    def __init__(self, R, r, mode, epsilon=10**(-6)):
        self.mode = mode
        self.epsilon = epsilon
        self.R = R
        self.r = r

    def project(self, z):
        self.model = pyo.ConcreteModel()

        # Variables
        if self.mode == 'multilabel-classification':
            self.model.z = pyo.Var(range(self.R.shape[0]), domain=pyo.Binary)
        else:
            self.model.z = pyo.Var(range(self.R.shape[0]))

        # Constraints
        @self.model.Constraint(range(self.R.shape[1]))
        def property(model, j):
            return model.z @ self.R[:,j] <= self.r[j]

        # Objective
        if self.mode == 'multilabel-classification':
            lpc = -np.log(np.maximum(self.epsilon, z))
            clpc = -np.log(np.maximum(self.epsilon, 1 - z))
            scale = max(np.max(lpc), np.max(clpc))
            lpc /= scale
            clpc /= scale
            expr = sum(self.model.z[i] * lpc[i] + (1 - self.model.z[i])*clpc[i] for i in range(self.R.shape[0]))
        else: 
            expr = sum((z[i] - self.model.z[i])**2 for i in range(self.R.shape[0]))
        self.model.objective = pyo.Objective(expr=expr, sense=pyo.minimize)

        # Solve
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model)

        z = np.array([self.model.z[j].value for j in range(self.R.shape[0])])

        return z

    def write_model(self, filename):
        self.model.write(filename, io_options = {"symbolic_solver_labels":True})
