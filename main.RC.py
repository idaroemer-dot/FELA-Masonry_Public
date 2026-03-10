from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri


#---------Topology---------# 
lenght = 1           # Total length   
height = 1    # Total height

nx = 4
ny = 2

a = (lenght/nx)/2     # Length of cell in x direction
b = (height/ny)/2     # Height of cell in y direction

base_nodes = np.array([
    [0, 0],
    [a, 0],
    [2*a, 0],
    [0.5*a, 0.5*b],
    [1.5*a, 0.5*b],
    [0, b],
    [a, b],
    [2*a, b],
    [0.5*a, 1.5*b],
    [1.5*a, 1.5*b],
    [0, 2*b],
    [a, 2*b],
    [2*a, 2*b]
])

base_elements = np.array([
    [1, 3, 7, 5, 4, 2],
    [1, 7, 11, 9, 6, 4],
    [3, 13, 7, 10, 5, 8],
    [7, 13, 11, 12, 9, 10]
]) - 1


# multiple cells
all_nodes = []
all_elements = []

node_dict = {}        # maps coordinate tuple -> global index
node_list = []        # final unique nodes

for i in range(nx):
    for j in range(ny):

        shift = np.array([2*a*i, 2*b*j])
        shifted_nodes = base_nodes + shift

        local_to_global = []

        # --- build global node numbering ---
        for node in shifted_nodes:
            key = tuple(np.round(node, 10))  # avoid float issues

            if key in node_dict:
                global_index = node_dict[key]
            else:
                global_index = len(node_list)
                node_dict[key] = global_index
                node_list.append(node)

            local_to_global.append(global_index)

        local_to_global = np.array(local_to_global)

        # --- build elements using mapped indices ---
        for el in base_elements:
            all_elements.append(local_to_global[el])

node_coordinates = np.array(node_list)
elements_topology = np.array(all_elements)

number_nodes = node_coordinates.shape[0]
number_elements = elements_topology.shape[0]

print("Number of nodes:", number_nodes)
print("Number of elements:", number_elements)

# materials: G[el] = [t, fc, Ax, fYx, Ay, fYy]
t = 0.2
fc = 20e6
Ax = 500e-6
fYx = 300e6
Ay = Ax
fYy = 300e6

#G = np.array([
#    [t, fc, Ax, fYx, Ay, fYy],
#    [t, fc, Ax, fYx, Ay, fYy],
#    [t, fc, Ax, fYx, Ay, fYy],
#    [t, fc, Ax, fYx, Ay, fYy]
#])

G_base = np.array([[t, fc, Ax, fYx, Ay, fYy]])
G = np.tile(G_base, (number_elements, 1))

# supports[i] = [node, direction]
tol = 1e-10
supports = []

for i, (x, y) in enumerate(node_coordinates):

    # Venstre side (symmetri): Ux = 0
    if abs(x - 0.0) < tol:
        supports.append([i, 0])

    # Højre side (rulle): Uy = 0
    if abs(x - lenght) < tol:
        supports.append([i, 1])

supports = np.array(supports, dtype=int)

#supports = np.array([
    # pinned supports (Ux = 0, Uy = 0)
#    [1, 0],
 #   [1, 1],
#    [2, 0],
 #   [2, 1],
  #  [3, 0],
#    [3, 1],
    # roller supports (Uy = 0)
  #  [943, 1],
  #  [1008, 1],
 #   [1009, 1]
#])

#supports[:, 0] -= 1

# loads[i] = [node, direction, value]
f=1
tol = 1e-10
loads = []

# find all nodes on top boundary
top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
             if abs(y - height) < tol]

# total load F_total
F_total = f

# distribute evenly
for i in top_nodes:
    loads.append([i, 1, -F_total / len(top_nodes)])

loads = np.array(loads, dtype=float)

print(loads)

#loads = np.array([
 #   [50, 1, -1/2*f],
 #   [51, 1, -1*f],
 #   [102, 1, -1*f],
 #   [103, 1, -1*f],
 #   [154, 1, -1*f],
 #   [155, 1, -1*f],
 #   [206, 1, -1*f],
 #   [207, 1, -1*f],
 #   [258, 1, 1*f],
 #   [259, 1, -1*f],
 #   [310, 1, -1*f],
 #   [311, 1, -1*f],
 #   [362, 1, -1*f],
 #   [363, 1, 1*f],
 #   [414, 1, -1*f],
 #   [415, 1, -1*f],
 #   [416, 1, -1/2*f]
#])

#loads = np.array([
#    [544, 1, -f],
 #   [545, 1, -f],
 #   [546, 1, -f],
#])

#loads[:, 0] -= 1

#Setup global mapping
from fem.assembly import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(node_coordinates, elements_topology)

#Plot geometry
from post.plot_topology import plot_topology
plotTop = plot_topology(node_coordinates, elements_topology, number_elements, number_nodes)

#Establish equilibrium matrix
from fem.equilibrium import global_equilibrium_matrix
global_equilibrium = global_equilibrium_matrix(node_coordinates, elements_topology, number_elements, number_equations, number_variabels, equations_per_element, variables_per_element)

#Set supports
from model.supports import setsup
number_sup, global_DOF_index_supports, global_equilibrium_reduced = setsup(supports, global_equilibrium)

#Set loads 
from model.loads import setload
number_load, global_load_vector, global_load_vector_reduced = setload(2*number_nodes, global_DOF_index_supports, loads)

#Set constraints
from fem.constrains_RC import setcon
Ab, blc, buc, C = setcon(number_elements, G)

#Optimize
from optimization.mosek_solver import solveopt
x, y, lambda_val = solveopt(number_elements, global_equilibrium_reduced, global_load_vector_reduced, Ab, blc, buc, C)

#Plot principal stresses
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-7)

#Plot displacements
from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes,1e-1)
