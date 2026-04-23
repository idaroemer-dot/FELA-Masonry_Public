from turtle import left

from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix

#---------Topology---------# 
L_wall = 0.990
overhang = 0.055
L_top = L_wall + 2 * overhang

h_wall = 1.000
h_top = 0.070

# Wall position inside the full top beam width
x_wall_min = overhang
x_wall_max = overhang + L_wall

#---------Mesh parameters---------#
# course mesh
# nx_total = 10         
# nx_wall = 9          

# medium mesh
nx_total = 20         
nx_wall = 18    

# fine mesh
# nx_total = 40         
# nx_wall = 36    

ny_wall = 20           
ny_beam = 1   

# Base cell
def base_cell(a, b):
    base_nodes = np.array([
        [0, 0],
        [a, 0],
        [2 * a, 0],
        [0.5 * a, 0.5 * b],
        [1.5 * a, 0.5 * b],
        [0, b],
        [a, b],
        [2 * a, b],
        [0.5 * a, 1.5 * b],
        [1.5 * a, 1.5 * b],
        [0, 2 * b],
        [a, 2 * b],
        [2 * a, 2 * b]
    ])

    base_elements = np.array([
        [1, 3, 7, 5, 4, 2],
        [1, 7, 11, 9, 6, 4],
        [3, 13, 7, 10, 5, 8],
        [7, 13, 11, 12, 9, 10]
    ]) - 1
    return base_nodes, base_elements

# 6-node elements
def make_block_mesh(x0, y0, width, height, nx, ny, node_dict, node_list):
    a = (width / nx) / 2.0
    b = (height / ny) / 2.0

    base_nodes, base_elements = base_cell(a, b)
    all_elements = []

    for i in range(nx):
        for j in range(ny):
            shift = np.array([x0 + 2.0 * a * i, y0 + 2.0 * b * j])
            shifted_nodes = base_nodes + shift

            local_to_global = []
            for node in shifted_nodes:
                key = tuple(np.round(node, 12))
                if key in node_dict:
                    g = node_dict[key]
                else:
                    g = len(node_list)
                    node_dict[key] = g
                    node_list.append(node)
                local_to_global.append(g)

            local_to_global = np.array(local_to_global, dtype=int)

            for el in base_elements:
                all_elements.append(local_to_global[el])

    return all_elements

# Mesh assembly
node_dict = {}
node_list = []

# Wall block: only over wall width
wall_elements = make_block_mesh(x0=x_wall_min,y0=0.0,width=L_wall,height=h_wall,nx=nx_wall,ny=ny_wall,node_dict=node_dict,node_list=node_list)

# Beam block: full width, one layer
beam_elements = make_block_mesh(x0=0.0,y0=h_wall,width=L_top,height=h_top,nx=nx_total,ny=ny_beam,node_dict=node_dict,node_list=node_list)

node_coordinates = np.array(node_list, dtype=float)
elements_topology = np.array(wall_elements + beam_elements, dtype=int)

number_nodes = node_coordinates.shape[0]
number_elements = elements_topology.shape[0]

print("Number of nodes:", number_nodes)
print("Number of elements:", number_elements)
print("y_min =", np.min(node_coordinates[:, 1]))
print("y_max =", np.max(node_coordinates[:, 1]))

#---------Material parameters---------#
# masonry case I
t_masonry = 0.100
nu = 1.0
nu_t = 0.6
nu_c = 0.7 - 1.87 / 200
#nu_t = 1.0
#nu_c = 1.0
xi = 0.5
omega_max = np.deg2rad(50.0)
fcm = 10.5e6 * nu_c

# joints
ftx = 0.25e6 * nu_t
fce = 10.5e6 * nu_c
fcl = fce
phi_l = np.arctan(0.75)

# units
fcs = 30e6 * nu_c
phi_s = np.arctan(1.0)

# Steel beam as stiff strip
t_steel = 0.200
fy_steel = 250e6

# Material assignment
# 0 = masonry
# 3 = steel beam
G = np.zeros((number_elements, 11))
mat_type = np.zeros(number_elements, dtype=int)

for el in range(number_elements):
    conn = elements_topology[el]
    xy = node_coordinates[conn]
    xc = np.mean(xy[:, 0])
    yc = np.mean(xy[:, 1])

    # Beam
    if yc >= h_wall - 1e-12:
        mat_type[el] = 3
        G[el, :] = np.array([
            t_steel, fy_steel, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    # Wall
    else:
        mat_type[el] = 0
        G[el, :] = np.array([
            t_masonry, fcm, ftx, nu, fcs, fce,
            xi, phi_s, fcl, phi_l, omega_max
        ])

print("Steel elements =", np.sum(mat_type == 3))
print("Masonry elements =", np.sum(mat_type == 0))

#---------Supports---------#
tol = 1e-10
supports = []

bottom_nodes = [
    i for i, (x, y) in enumerate(node_coordinates)
    if abs(y - 0.0) < tol and (x_wall_min - tol <= x <= x_wall_max + tol)
]
bottom_nodes = sorted(bottom_nodes, key=lambda i: node_coordinates[i, 0])

for i in bottom_nodes:
    supports.append([i, 1])   # Uy = 0
    supports.append([i, 0])   # Ux = 0

supports = np.array(supports, dtype=int)

print("Supports:\n", supports)
print("Number of bottom nodes =", len(bottom_nodes))

# loads[i] = [node, direction, value]
Fv_total = 105.0e3   # [N]
Fh_ref_total = -1.0   # [N]

loads_const = []
loads_var = []

y_top = h_wall + h_top

top_nodes = [
    i for i, (x, y) in enumerate(node_coordinates)
    if abs(y - y_top) < tol
]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

top_nodes_load = [
    i for i in top_nodes
    if x_wall_min - tol <= node_coordinates[i, 0] <= x_wall_max + tol
]
top_nodes_load = sorted(top_nodes_load, key=lambda i: node_coordinates[i, 0])

top_x = np.array([node_coordinates[i, 0] for i in top_nodes_load])

tributary = np.zeros(len(top_nodes_load))
for k in range(len(top_nodes_load)):
    if k == 0:
        tributary[k] = 0.5 * (top_x[k + 1] - top_x[k])
    elif k == len(top_nodes_load) - 1:
        tributary[k] = 0.5 * (top_x[k] - top_x[k - 1])
    else:
        tributary[k] = 0.5 * (top_x[k + 1] - top_x[k - 1])

weights = tributary / np.sum(tributary)

for k, i in enumerate(top_nodes_load):
    loads_const.append([i, 1, -Fv_total * weights[k]])
    loads_var.append([i, 0, Fh_ref_total * weights[k]])

loads_const = np.array(loads_const, dtype=float)
loads_var = np.array(loads_var, dtype=float)

print("Sum vertical load [kN] =", np.sum(loads_const[:, 2]) / 1000.0)
print("Sum horizontal reference load [N] =", np.sum(loads_var[:, 2]))


#------Global setup------#
from fem.assembly import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(
    node_coordinates, elements_topology
)

from post.plot_topology import plot_topology
plotTop = plot_topology(
    node_coordinates, elements_topology, number_elements, number_nodes
)

from fem.equilibrium import global_equilibrium_matrix
global_equilibrium = global_equilibrium_matrix(node_coordinates,elements_topology,number_elements,number_equations,number_variabels,equations_per_element,variables_per_element)

from model.supports import setsup
number_sup, global_DOF_index_supports, global_equilibrium_reduced = setsup(supports, global_equilibrium)

from model.loads import setload
number_load_const, global_load_vector_const, global_load_vector_reduced_const = setload(2 * number_nodes, global_DOF_index_supports, loads_const)
number_load_var, global_load_vector_var, global_load_vector_reduced_var = setload(2 * number_nodes, global_DOF_index_supports, loads_var)

from fem.constrains_mixed import setcon_mixed
Ab, blc, buc, C = setcon_mixed(number_elements, G, mat_type)

# Optimization
from optimization.mosek_solver import solveopt
x, alpha, y, lambda_val = solveopt(number_elements,global_equilibrium_reduced,global_load_vector_reduced_const,global_load_vector_reduced_var,Ab, blc, buc, C)

sx_all = []
sy_all = []
tau_all = []

for el in range(number_elements):
    for gp in range(3):
        idx = 9 * el + 3 * gp
        sx_all.append(x[idx + 0])
        sy_all.append(x[idx + 1])
        tau_all.append(x[idx + 2])

print("lambda =", lambda_val)
print("min sy =", np.min(sy_all))
print("max sy =", np.max(sy_all))


# Collapse load
Fh_collapse_full = lambda_val * Fh_ref_total

print("Vertical load on full model [kN] =", -np.sum(loads_const[:, 2]) / 1000)
print("Horizontal collapse load, full wall [kN] =", Fh_collapse_full / 1000)


# Post-processing
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-6)

from post.plot_displacements import plotDof
plotDof(node_coordinates,elements_topology,y,global_DOF_index_supports,number_elements,number_nodes,1e-1)