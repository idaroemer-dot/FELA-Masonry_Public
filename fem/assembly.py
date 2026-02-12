import numpy as np

def setup_global_mapping(node_coordinates, elements_topology):
    number_nodes = node_coordinates.shape[0]
    number_elements = elements_topology.shape[0]

    # 9 variables per element
    variables_per_element = np.zeros((number_elements, 9), dtype=int)  # Matrix to hold variable indices for each element
    for el in range(number_elements):
        variables_per_element[el, :] = el * 9 + np.arange(9)   
    number_variabels = 9 * number_elements

    # 2 equilibrium equations per node
    number_equations = 2 * number_nodes

    equations_per_element = np.zeros((number_elements, 12), dtype=int) # Matrix to hold equation indices for each element
    for el in range(number_elements):
        for no in range(6):
            equations_per_element[el, 2*no]     = 2 * elements_topology[el, no]
            equations_per_element[el, 2*no + 1] = 2 * elements_topology[el, no] + 1

    return number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element
