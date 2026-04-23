from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri

# --------------------------------------------------
# Geometry: half model due to symmetry
# --------------------------------------------------
L = 3.600
h = 2.000

length = L / 2.0          # half wall
height = h

nx = 12                   # about half of previous full model
ny = 8

a = (length / nx) / 2
b = (height / ny) / 2

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

# --------------------------------------------------
# Mesh assembly
# --------------------------------------------------
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
# --------------------------------------------------
# Base strengths
# --------------------------------------------------
ftx_web_base = 0.28e6
fcx_web_base = 1.87e6
fcy_web_base = 7.61e6

# flange strengths from text
ftx_flange_base = 0.68e6
fcx_flange_base = 9.5e6

# reduction factors
nu = 1.0
nu_t = 0.6
nu_c = 0.7 - 1.87/200
xi = 0.5
omega_max = np.deg2rad(50.0)

phi_l = np.arctan(0.75)
phi_s = np.arctan(1.0)

# web properties
ftx_web = ftx_web_base * nu_t
fcm_web = fcx_web_base * nu_c
fce_web = fcm_web
fcl_web = fcm_web
fcs_web = 30e6 * nu_c

# flange properties
ftx_flange = ftx_flange_base * nu_t
fcm_flange = fcx_flange_base * nu_c
fce_flange = fcm_flange
fcl_flange = fcm_flange
fcs_flange = fcs_web

# Thickness variation (flange zone)
t_web = 0.150
t_flange = 0.600

G = np.zeros((number_elements, 11))

for el in range(number_elements):
    conn = elements_topology[el]
    xy = node_coordinates[conn]
    xc = np.mean(xy[:, 0])

    # default = web
    t_el   = t_web
    fcm_el = fcm_web
    ftx_el = ftx_web
    fcs_el = fcs_web
    fce_el = fce_web
    fcl_el = fcl_web

    # flange zone
    if xc <= 0.150:
        t_el   = t_flange
        fcm_el = fcm_flange
        ftx_el = ftx_flange
        fcs_el = fcs_flange
        fce_el = fce_flange
        fcl_el = fcl_flange

    G[el, :] = np.array([
        t_el, fcm_el, ftx_el, nu, fcs_el, fce_el,
        xi, phi_s, fcl_el, phi_l, omega_max
    ])

print("min thickness =", np.min(G[:,0]))
print("max thickness =", np.max(G[:,0]))
print("number of flange elements =", np.sum(G[:,0] > t_web))

# Supports
tol = 1e-10
supports = []

# bottom support: Uy = 0 only
bottom_nodes = [i for i, (x, y) in enumerate(node_coordinates)
                if abs(y - 0.0) < tol]
bottom_nodes = sorted(bottom_nodes, key=lambda i: node_coordinates[i, 0])

for i in bottom_nodes:
    supports.append([i, 1])   # Uy = 0
    supports.append([i, 0])   # Ux = 0

# symmetry boundary at x = length -> Ux = 0
sym_nodes = [i for i, (x, y) in enumerate(node_coordinates)
             if abs(x - length) < tol]
sym_nodes = sorted(sym_nodes, key=lambda i: node_coordinates[i, 1])

for i in sym_nodes:
    supports.append([i, 0])   # Ux = 0 on symmetry line

supports = np.array(supports, dtype=int)
print("Supports:\n", supports)


# Loads
Fv_total = 415.0e3 / 2.0   # [N]
Fh_ref_total = 1.0         # [N]

loads_const = []
loads_var = []

top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
             if abs(y - height) < tol]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

top_x = np.array([node_coordinates[i, 0] for i in top_nodes])

tributary = np.zeros(len(top_nodes))
for k in range(len(top_nodes)):
    if k == 0:
        tributary[k] = 0.5 * (top_x[k + 1] - top_x[k])
    elif k == len(top_nodes) - 1:
        tributary[k] = 0.5 * (top_x[k] - top_x[k - 1])
    else:
        tributary[k] = 0.5 * (top_x[k + 1] - top_x[k - 1])

weights = tributary / np.sum(tributary)

for k, i in enumerate(top_nodes):
    loads_const.append([i, 1, -Fv_total * weights[k]])
    loads_var.append([i, 0, Fh_ref_total * weights[k]])

loads_const = np.array(loads_const, dtype=float)
loads_var = np.array(loads_var, dtype=float)

print("Loads (constant):\n", loads_const)
print("Loads (variable):\n", loads_var)
print("Sum vertical load [kN] =", np.sum(loads_const[:, 2]) / 1000)
print("Sum horizontal reference load [N] =", np.sum(loads_var[:, 2]))


# Global setup
from fem.assembly import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(
    node_coordinates, elements_topology
)

from post.plot_topology import plot_topology
plotTop = plot_topology(node_coordinates, elements_topology, number_elements, number_nodes)

from fem.equilibrium import global_equilibrium_matrix
global_equilibrium = global_equilibrium_matrix(
    node_coordinates, elements_topology, number_elements, number_equations,
    number_variabels, equations_per_element, variables_per_element
)

from model.supports import setsup
number_sup, global_DOF_index_supports, global_equilibrium_reduced = setsup(
    supports, global_equilibrium
)

from model.loads import setload
number_load_const, global_load_vector_const, global_load_vector_reduced_const = setload(
    2 * number_nodes, global_DOF_index_supports, loads_const
)
number_load_var, global_load_vector_var, global_load_vector_reduced_var = setload(
    2 * number_nodes, global_DOF_index_supports, loads_var
)

from fem.constrains_masonry import setcon_masonry
Ab, blc, buc, C = setcon_masonry(number_elements, G)

# Optimization
from optimization.mosek_solver import solveopt
x, alpha, y_tied, lambda_val = solveopt(
    number_elements,
    global_equilibrium_reduced,
    global_load_vector_reduced_const,
    global_load_vector_reduced_var,
    Ab, blc, buc, C
)

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
Fh_collapse_half = lambda_val * Fh_ref_total
Fh_collapse_full = 2.0 * Fh_collapse_half

print("Vertical load on half model [kN] =", -np.sum(loads_const[:, 2]) / 1000)
print("Horizontal collapse load, half model [kN] =", Fh_collapse_half / 1000)
print("Horizontal collapse load, full wall equivalent [kN] =", Fh_collapse_full / 1000)

sigma_eq = Fh_collapse_full / (t_web * h)
print("Equivalent horizontal stress [MPa] =", sigma_eq / 1e6)


# Post-processing
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-6)

from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y_tied, global_DOF_index_supports,
        number_elements, number_nodes, 1e-2)