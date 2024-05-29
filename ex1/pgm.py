# using pgm py solve the following problem

# In the past year there was a lion spotted in Kleinmachnow, Berlin. Given the
# following random variables
# Z = Zoo misses a lion
# L = Lion present
# B = Boar present
# S = Lion reported
# and probabilities
# P(Z = 1) = 10−3
# P(B = 1) = 10−1
# P(L = 1|Z = 1) = 0.95
# P(L = 1|Z = 0) = 10−4


import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianModel([('Z', 'L'), ('L', 'S'), ('B', 'S')])
# Defining individual CPDs.
cpd_z = TabularCPD(variable='Z', variable_card=2, values=[[0.999], [0.001]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.9], [0.1]])
cpd_l = TabularCPD(variable='L', variable_card=2, values=[[0.9999, 0.0001], [0.05, 0.95]],
                   evidence=['Z'], evidence_card=[2])

# Suppose the false positive rate for reporting a lion is l = 10−3; the rate for reporting a
# lion in the case of a boar αb = 0.05 and reporting a lion when a lion is present αL = 0.95.
# (a) Implement and plot the corresponding directed graphical model in a jupyter notebook (for a hint see Figure). You will want to explore igraph or networkx. For
# networkx checkout d-seperation.

cpd_s = TabularCPD(variable='S', variable_card=2,
                     values=[[1, 0.05, 0.95, 0.95],
                            [0, 0.95, 0.05, 0.05]],
                     evidence=['L', 'B'], evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_z, cpd_b, cpd_l, cpd_s)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly
# defined and sum to 1.
model.check_model()

# Doing exact inference using Variable Elimination
infer = VariableElimination(model)
posterior = infer.query(['L'], evidence={'S': 1})
print(posterior['L'])

# Doing exact inference using Variable Elimination
posterior = infer.query(['L'], evidence={'S': 1, 'B': 1})
print(posterior['L'])

