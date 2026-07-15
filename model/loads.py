import numpy as np

def setload(number_equations, global_DOF_index_supports, loads):
    global_load_vector = np.zeros(number_equations)
    number_load = loads.shape[0]

    global_DOF_index_load = 2*loads[:, 0] + loads[:, 1]
    global_DOF_index_load = global_DOF_index_load.astype(int)
    
    np.add.at(global_load_vector, global_DOF_index_load, loads[:, 2])

    global_load_vector_reduced = np.delete(global_load_vector, global_DOF_index_supports)

    return number_load, global_load_vector, global_load_vector_reduced


def element_area(node_coordinates, element_nodes):
    x1, y1 = node_coordinates[element_nodes[0], :2]
    x2, y2 = node_coordinates[element_nodes[1], :2]
    x3, y3 = node_coordinates[element_nodes[2], :2]

    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return area

def self_load_weight(node_coordinates, element_topology, thickness, density=1800.0, g=9.81):
    loads = []
    for element_nodes in element_topology:
        area = element_area(node_coordinates, element_nodes)
        weight = area * thickness * density * g

        midside_nodes = element_nodes[3:6]
        for node in midside_nodes:
            loads.append([node, 1, -weight / 3.0])  # distribute weight equally among the three nodes

    return np.array(loads, dtype=float)

def add_self_loads(loads_const, node_coordinates, element_topology, thickness, density=1800.0, g=9.81):
    loads_const = np.asarray(loads_const, dtype=float)
    loads_self = self_load_weight(node_coordinates, element_topology, thickness, density, g)

    if loads_const.size == 0:
        return loads_self

    return np.vstack((loads_const, loads_self))
