from turtle import left

from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri
#---------Topology---------# 
lenght = 0.99
height = 1.00

nx = 4
ny = 4

a = (lenght/nx)/2
b = (height/ny)/2

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

cell_w = 2*a   
cell_h = 2*b   

# openings: (x_left, x_right, y_bottom, y_top)
openings_xy = []

def cell_is_in_opening(i, j):
    x0 = i * cell_w
    x1 = (i + 1) * cell_w
    y0 = j * cell_h
    y1 = (j + 1) * cell_h

    for ox0, ox1, oy0, oy1 in openings_xy:
        if x0 >= ox0 and x1 <= ox1 and y0 >= oy0 and y1 <= oy1:
            return True
    return False

all_elements = []
node_dict = {}
node_list = []

for i in range(nx):
    for j in range(ny):

        if cell_is_in_opening(i, j):
            continue

        shift = np.array([2*a*i, 2*b*j])
        shifted_nodes = base_nodes + shift

        local_to_global = []

        for node in shifted_nodes:
            key = tuple(np.round(node, 10))

            if key in node_dict:
                global_index = node_dict[key]
            else:
                global_index = len(node_list)
                node_dict[key] = global_index
                node_list.append(node)

            local_to_global.append(global_index)

        local_to_global = np.array(local_to_global)

        for el in base_elements:
            all_elements.append(local_to_global[el])

node_coordinates = np.array(node_list)
elements_topology = np.array(all_elements)

number_nodes = node_coordinates.shape[0]
number_elements = elements_topology.shape[0]

print("Number of nodes:", number_nodes)
print("Number of elements:", number_elements)

# materials: G[el] = [t, ctau, cn, mu]
# masonry
t   = 0.10   #[m] helstensvæg
fcm = 10.5e6  #[Pa]
ftx = 0.25e6  #[Pa]
nu = 0.8
fcs = 0.5*fcm
fce = fcs
xi = 0.5
phi_s = np.deg2rad(30)
fcl = 2.0e6
phi_l = np.deg2rad(37)
omega_max = np.deg2rad(45.0) 

G_base = np.array([[t, fcm, ftx, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max]])
G = np.tile(G_base, (number_elements, 1))

# supports[i] = [node, direction]
tol = 1e-10
supports = []

for i, (x, y) in enumerate(node_coordinates):
    if abs(y - 0.0) < tol:
        supports.append([i, 1])   # Uy = 0 on bottom
       # supports.append([i, 0])   # Ux = 0 on bottom

# fix one x-DOF to avoid rigid body motion
for i, (x, y) in enumerate(node_coordinates):
    if abs(x - 0.0) < tol and abs(y - 0.0) < tol:
        supports.append([i, 0])   # Ux = 0 at bottom-left corner
        break        

supports = np.array(supports, dtype=int)

print(supports)

# loads[i] = [node, direction, value]
# Pa, choose 0.30e6, 1.21e6, or 2.12e6
#qv = 0.30e6 * t   # N/m vertical compression line load
qv = 2
qh_ref = 1       # N/m unit horizontal reference shear load

loads = []

top_nodes = [i for i, (x, y) in enumerate(node_coordinates) if abs(y - height) < tol]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])
top_spacing = lenght / (len(top_nodes) - 1)

for k, i in enumerate(top_nodes):
    w = 0.5 if (k == 0 or k == len(top_nodes)-1) else 1.0

    # dead vertical load
    loads.append([i, 1, -qv * top_spacing * w])

    # reference horizontal shear load
    loads.append([i, 0,  qh_ref * top_spacing * w])

loads = np.array(loads, dtype=float)

print(loads)
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
x, alpha, y, lambda_val = solveopt(number_elements, global_equilibrium_reduced, global_load_vector_reduced, Ab, blc, buc, C)

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


# collapse load corresponding to q
p_collapse = lambda_val * qh_ref 
print("Collapse load intensity p* =", p_collapse/ (t* 1e6), "MPa")
print("Collapse load intensity p* =", p_collapse/1000, "kN/m")
print("Collapse load intensity p* =", 2 * p_collapse * lenght/1000, "kN")


#Plot principal stresses
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-4)

#Plot displacements 
from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes,1e-1)


