import matplotlib.pyplot as plt
import numpy as np

def plot_topology(node_coordinates, elements_topology, number_elements, number_nodes):
    # plot geometry
    plt.figure(1)
    plt.title('Element topologi')
    plt.axis('equal')
    local_dof = [0, 5, 1, 3, 2, 4, 0] # local dof indices for plotting element edges

    for el in range(number_elements):
        plt.plot(node_coordinates[elements_topology[el, local_dof], 0], node_coordinates[elements_topology[el, local_dof], 1], 'b-')

    for no in range(number_nodes): # write node numbers
       plt.text(node_coordinates[no, 0], node_coordinates[no, 1], str(no + 1),
                color='blue', backgroundcolor=(0.7, 0.7, 0.7))

    # for el in range(number_elements): # write element numbers
    #    xp = np.mean(node_coordinates[elements_topology[el, :], :], axis=0)
    #    plt.text(xp[0], xp[1], str(el + 1), color='black')

    plt.show()