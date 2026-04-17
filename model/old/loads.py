import numpy as np

def setload(number_equations, global_DOF_index_supports, loads):
    global_load_vector = np.zeros(number_equations)
    number_load = loads.shape[0]

    global_DOF_index_load = 2*loads[:, 0] + loads[:, 1]
    global_DOF_index_load = global_DOF_index_load.astype(int)

    global_load_vector[global_DOF_index_load] += loads[:, 2] # accumulate loads if multiple loads are applied to the same DOF

    global_load_vector_reduced = np.delete(global_load_vector, global_DOF_index_supports)

    return number_load, global_load_vector, global_load_vector_reduced
