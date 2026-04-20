from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri

#---------Topology---------# 
L = 0.757
h = 0.457

lenght = L/2      # half beam due to symmetry
height = h

nx = 8
ny = 10

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


all_elements = []
node_dict = {}
node_list = []

for i in range(nx):
    for j in range(ny):

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
t   = 0.054                         #[m]  thickness
nu = 1                              # effectiveness factor for shear strength
nu_t = 0.6                           #effectiveness factor for tensile strength (2013 Portioli)
nu_c = 0.7 - 8.60/200                #effectiveness factor for compressive strength (2013 Portioli)
xi = 0.5                            # relation between unit and head-joint area
omega_max = np.deg2rad(50.0)        # maximum angle related to staircase failure mechanism
fcm = 8.60e6 * nu_c                   #[Pa] compressive strength

# joints
ftx = 0.29e6 * nu_t                  #[Pa] tensile strength in x direction
fce =  8.60e6 * nu_c                       #[Pa] compressive strength for the head-joint
fcl =  fce                     #[Pa] compressive strength for the bed joint
phi_l = np.arctan(0.75)           # friction angle of the bed joint 

# units
fcs = 30e6 * nu_c                      #[Pa] compressive strength for the unit
phi_s = np.arctan(1.0)      # [rad] friction angle of the unit

G_base = np.array([[t, fcm, ftx, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max]])
G = np.tile(G_base, (number_elements, 1))

# supports[i] = [node, direction]
tol = 1e-10
support_length = 0.188
x_support_start = lenght - support_length

supports = []

for i, (x, y) in enumerate(node_coordinates):

    # symmetry boundary at x = 0
    if abs(x - 0.0) < tol:
        supports.append([i, 0])   # Ux = 0 only

    # bottom support over the right support region
    if abs(y - 0.0) < tol and x >= (x_support_start - tol):
        supports.append([i, 1])   # Uy = 0
        supports.append([i, 0])   # Ux = 0   (rigid support)
supports = np.array(supports, dtype=int)
print("Supports:\n", supports)

# loads[i] = [node, direction, magnitude]
q = 1.0   # reference line load [N/m]
tol = 1e-10
loads = []

load_length = 0.381 / 2   # half of steel beam length

# top nodes
top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
             if abs(y - height) < tol]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

# loop over top edge segments and load only those within x in [0, load_length]
for k in range(len(top_nodes) - 1):
    n1 = top_nodes[k]
    n2 = top_nodes[k + 1]

    x1, y1 = node_coordinates[n1]
    x2, y2 = node_coordinates[n2]

    # only include segments fully inside the loaded zone
    if x2 <= load_length + tol:
        Le = x2 - x1
        fnod = q * Le / 2.0

        loads.append([n1, 1, -fnod])
        loads.append([n2, 1, -fnod])

loads = np.array(loads, dtype=float)
print("Loads:\n", loads)
print("Total applied half-model load =", np.sum(loads[:, 2]))
# --------------------------------------------------
# Global setup
# --------------------------------------------------
from fem.assembly import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(node_coordinates, elements_topology)

from post.plot_topology import plot_topology
plotTop = plot_topology(node_coordinates, elements_topology, number_elements, number_nodes)

from fem.equilibrium import global_equilibrium_matrix
global_equilibrium = global_equilibrium_matrix(
    node_coordinates,
    elements_topology,
    number_elements,
    number_equations,
    number_variabels,
    equations_per_element,
    variables_per_element
)

from model.supports import setsup
number_sup, global_DOF_index_supports, global_equilibrium_reduced = setsup(supports, global_equilibrium)

from model.loads import setload
number_load, global_load_vector, global_load_vector_reduced = setload(2*number_nodes, global_DOF_index_supports, loads)

from fem.constrains_RC import setcon
Ab, blc, buc, C = setcon(number_elements, G)

# --------------------------------------------------
# Optimization
# --------------------------------------------------
from optimization.mosek_solver_RC import solveopt
x, y, lambda_val = solveopt(
    number_elements,
    global_equilibrium_reduced,
    global_load_vector_reduced,
    Ab, blc, buc, C
)

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
p_collapse = lambda_val * q   # [N/m]

P_half = p_collapse * load_length
P_full = 2 * P_half

print("Collapse load line intensity q* =", p_collapse/1000, "kN/m")
print("Collapse load on half model =", P_half/1000, "kN")
print("Collapse load on full beam =", P_full/1000, "kN")
# --------------------------------------------------
# Post-processing
# --------------------------------------------------
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-7)

from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes, 1e-3)