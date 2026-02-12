import numpy as np

# connectivity helper
cy = np.array([
    [0, 1, 2],   # [i, j, k] 
    [1, 2, 0],   # [i, j, k]
    [2, 0, 1]    # [i, j, k]
])

# compute element equilibrium matrix

def element_equilibrium_matrix(node_coordinates):
    number_equations, number_variabels = 12, 9
    h = np.zeros((number_equations, number_variabels))

    a = np.zeros(3)
    b = np.zeros(3)

    # geometry
    for i in range(3):
        j = cy[1, i]
        k = cy[2, i]
        a[i] = node_coordinates[k, 0] - node_coordinates[j, 0]
        b[i] = node_coordinates[k, 1] - node_coordinates[j, 1]

    P = []
    for i in range(3):
        P.append(np.array([
            [ b[i],   0.0, -a[i]],
            [ 0.0, -a[i],  b[i]]
        ]))

    O23 = np.zeros((2, 3))

    element_equilibrium = np.block([
        [-P[0],        O23,        O23],
        [ O23,       -P[1],        O23],
        [ O23,         O23,      -P[2]],
        [ P[0],   P[0]-P[2],  P[0]-P[1]],
        [P[1]-P[2],    P[1],  P[1]-P[0]],
        [P[2]-P[1], P[2]-P[0],     P[2]]
    ]) / 6.0

    return element_equilibrium

# establish global equilibrium matrix
def global_equilibrium_matrix(node_coordinates, elements_topology, number_elements, number_equations, number_variabels, equations_per_element, variables_per_element):
    global_equilibrium = np.zeros((number_equations, number_variabels))

    for el in range(number_elements):
        element_equilibrium = element_equilibrium_matrix(node_coordinates[elements_topology[el, :], :])
        global_equilibrium[np.ix_(equations_per_element[el], variables_per_element[el])] += element_equilibrium

    return global_equilibrium

