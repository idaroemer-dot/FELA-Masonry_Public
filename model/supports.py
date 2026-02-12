import numpy as np

def setsup(supports, global_equilibrium):
    number_sup = supports.shape[0]
    global_DOF_index_supports = 2*supports[:, 0] + supports[:, 1]
    global_DOF_index_supports = global_DOF_index_supports.astype(int)
    global_equilibrium_reduced = np.delete(global_equilibrium, global_DOF_index_supports, axis=0)
    return number_sup, global_DOF_index_supports, global_equilibrium_reduced