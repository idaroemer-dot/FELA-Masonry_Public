import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# topology
lenght = 0.757      
height = 0.457

nx = 16
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

# mesh assembly
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

# materials: G[el] = [t, ctau, cn, mu....]
# masonry
t   = 0.054                             #[m]  thickness
nu_t = 0.6                              # effectiveness factor for tensile strength 
nu_c = 0.7 - 8.60/200                   # effectiveness factor for compressive strength
xi = 0.5                                # relation between unit and head-joint area
omega_max = np.deg2rad(24.0)            # maximum angle related to staircase failure mechanism
fcm = 8.60e6 * nu_c                     #[Pa] compressive strength

# joints
ftx = 0.29e6 * nu_t                     #[Pa] tensile strength in x direction
fce =  8.60e6 * nu_c                    #[Pa] compressive strength for the head-joint
fcl =  fce                              #[Pa] compressive strength for the bed joint
phi_l = np.arctan(0.75)                 #[rad] friction "angle" of the bed joint 
#phi_l = np.rad2deg(30) 

# units
fcs = 30e6 * nu_c                       #[Pa] compressive strength for the unit
fts = 0.29e6 * nu_t                     #[Pa] tensile strength for the unit
phi_s = np.arctan(1.0)                  #[rad] friction angle of the unit

G_base = np.array([[t, fcm, ftx, nu_c, fcs, fts, fce, xi, phi_s, fcl, phi_l, omega_max]])
G = np.tile(G_base, (number_elements, 1))

# supports[i] = [node, direction]
tol = 1e-10
support_length = 0.188

x_left_end  = support_length
x_right_start = lenght - support_length

supports = []

for i, (x, y) in enumerate(node_coordinates):
    if abs(y - 0.0) < tol:

        # left support region
        if x <= x_left_end + tol:
            supports.append([i, 1])   # Uy = 0
            supports.append([i, 0])   # Ux = 0

        # right support region
        elif x >= x_right_start - tol:
            supports.append([i, 1])   # Uy = 0
            supports.append([i, 0])   # Ux = 0

supports = np.array(supports, dtype=int)
print("Supports:\n", supports)


# loads[i] = [node, direction, magnitude]
q = 1.0
tol = 1e-10
Ls = 0.381

x_start = 0.1905
x_end   = 0.5715

# top nodes
top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
             if abs(y - height) < tol]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

# collect nodal loads in a dictionary
nodal_loads = {}

for k in range(len(top_nodes) - 1):
    n1 = top_nodes[k]
    n2 = top_nodes[k + 1]

    x1, y1 = node_coordinates[n1]
    x2, y2 = node_coordinates[n2]

    # overlap between segment and loaded interval
    xa = max(x1, x_start)
    xb = min(x2, x_end)

    if xb > xa:
        Le = xb - xa
        fnod = q * Le / 2.0

        nodal_loads[n1] = nodal_loads.get(n1, 0.0) - fnod
        nodal_loads[n2] = nodal_loads.get(n2, 0.0) - fnod

# convert to load array
loads = np.array([[node, 1, val] for node, val in sorted(nodal_loads.items())],
                 dtype=float)

print("Loads:\n", loads)
print("Total applied load =", np.sum(loads[:, 2]))

# global setup
from fem.assembly import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(node_coordinates, elements_topology)

from post.plot_topology import plot_topology
plotTop = plot_topology(node_coordinates, elements_topology, number_elements, number_nodes)

from fem.equilibrium import global_equilibrium_matrix
global_equilibrium = global_equilibrium_matrix(node_coordinates,elements_topology,number_elements,number_equations,number_variabels,equations_per_element,variables_per_element)

from model.supports import setsup
number_sup, global_DOF_index_supports, global_equilibrium_reduced = setsup(supports, global_equilibrium)

from model.loads import setload
number_load, global_load_vector, global_load_vector_reduced = setload(2*number_nodes, global_DOF_index_supports, loads)

from fem.constrains_masonry import setcon_masonry
Ab, blc, buc, C = setcon_masonry(number_elements, G)

# optimization
from optimization.mosek_solver_RC import solveopt
x, y, lambda_val = solveopt(number_elements,global_equilibrium_reduced,global_load_vector_reduced,Ab, blc, buc, C)

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
P_total = p_collapse * Ls     # total load over loaded length
print("Collapse load line intensity q* =", p_collapse/1000, "kN/m")
print("Total collapse load on beam =", P_total/1000, "kN")

# post-processing
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-7)

from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes, 1e-3)