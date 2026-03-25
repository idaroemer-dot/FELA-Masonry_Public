from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri


#---------Topology---------# 
lenght = 1    # Total length   
height = 1    # Total height

nx = 10
ny = 10

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

# materials: G[el] = [t, ctau, cn, mu]
# t   = 0.2   #[m]
# fcm = 20e6  #[Pa]
# ftx = 300e6  #[Pa]
# nu = 0.2

t   = 0.228   #[m] helstensvæg
fcm = 2e6  #[Pa]
ftx = 0.3e6  #[Pa]
nu = 0.2
G_base = np.array([[t, fcm, ftx, nu]])
G = np.tile(G_base, (number_elements, 1))

# supports[i] = [node, direction]
tol = 1e-10
supports = []

# for i, (x, y) in enumerate(node_coordinates):
#     if abs(y - 0.0) < tol:
#         supports.append([i, 1])   # Uy = 0 on bottom

# # fix one x-DOF to avoid rigid body motion
# for i, (x, y) in enumerate(node_coordinates):
#     if abs(x - 0.0) < tol and abs(y - 0.0) < tol:
#         supports.append([i, 0])   # Ux = 0 at bottom-left corner
#         break

for i, (x, y) in enumerate(node_coordinates):

    # Left side (symmetri): Ux = 0
    if abs(x - 0.0) < tol:
        supports.append([i, 0])

    # Right side (rulle): Uy = 0
    if abs(x - lenght) < tol:
        supports.append([i, 1])

supports = np.array(supports, dtype=int)


# supports = np.array([
#     #pinned supports (Ux = 0, Uy = 0)
#    [1, 0],
#    [1, 1],
#    [2, 0],
#    [2, 1],
#    [3, 0],
#    [3, 1],

#     #roller supports (Uy = 0)
#     [215, 1],
#     [248, 1],
#     [249, 1]
# ])

# supports[:, 0] -= 1

print(supports)

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


# q = 1
# lam_ref = 0.35
# p = lam_ref * q * lenght / height   

# tol = 1e-10
# loads = []

# # ---- top boundary nodes ----
# top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
#              if abs(y - height) < tol]
# top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

# # ---- left boundary nodes ----
# left_nodes = [i for i, (x, y) in enumerate(node_coordinates)
#               if abs(x - 0.0) < tol
#               and y >= 0.8*height]
# left_nodes = sorted(left_nodes, key=lambda i: node_coordinates[i, 1])

# # nodal forces from line loads using trapezoidal rule
# top_spacing = lenght / (len(top_nodes) - 1)
# left_spacing = height / (len(left_nodes) - 1)

# # top load q downward
# for k, i in enumerate(top_nodes):
#     weight = 0.5 if (k == 0 or k == len(top_nodes)-1) else 1.0
#     loads.append([i, 1, -q * top_spacing * weight])

# left load p to the right
# for k, i in enumerate(left_nodes):
#     weight = 0.5 if (k == 0 or k == len(left_nodes)-1) else 1.0
#     loads.append([i, 0, p * left_spacing * weight])


# # loads[i] = [node, direction, value]
# f=100

# # find all nodes on top boundary
# top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
#              if abs(y - height) < tol]

# # total load F_total
# F_total = f

# # distribute evenly
# for i in top_nodes:
#     loads.append([i, 1, -F_total / len(top_nodes)])

# loads = np.array(loads, dtype=float)

# # total load F_total
# F_total = f

# # distribute evenly
# for i in top_nodes:
#     loads.append([i, 1, -F_total / len(top_nodes)])

loads = np.array(loads, dtype=float)

# loads = np.array([
#    [21, 1, -f1],
#    [22, 1, -f1],
#    [23, 1, -f1],
#    [21, 0, f2],
#    [16, 0, f2],
#    [11, 0, f2]
# ])

# loads[:, 0] -= 1

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
from fem.constrains_masonry import setcon_masonry
Ab, blc, buc, C = setcon_masonry(number_elements, G)

#Optimize
from optimization.mosek_solver import solveopt
x, y, lambda_val = solveopt(number_elements, global_equilibrium_reduced, global_load_vector_reduced, Ab, blc, buc, C)

sx_all = []
sy_all = []
tau_all = []

for el in range(number_elements):
    for gp in range(3):
        idx = 9*el + 3*gp
        sx_all.append(x[idx+0])
        sy_all.append(x[idx+1])
        tau_all.append(x[idx+2])

print("lambda =", lambda_val)
print("min sy =", np.min(sy_all))
print("max sy =", np.max(sy_all))


#Plot principal stresses
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-6)

#Plot displacements 
from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes,1e-1)
