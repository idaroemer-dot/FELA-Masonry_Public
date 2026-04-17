from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri

# --------------------------------------------------
# Geometry: half model of the deep beam
# --------------------------------------------------
L = 0.757
h = 0.457

lenght = L/2      # half beam due to symmetry
height = h

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
# masonry
t   = 0.054                     #[m]  thickness
fcm = 8.60e6                    #[Pa] compressive strength
ftx = 0.29e6                    #[Pa] tensile strength in x direction
ftm = ftx                       #[Pa] tensile strength
nu = 0.8                        # effectiveness factor for shear strength
fcs = fcm                       #[Pa] compressive strength for the unit
fce = fcs                       #[Pa] compressive strength for the head-joint
xi = 0.5                        # relation between unit and head-joint area
phi_s = np.deg2rad(36.87)        # friction angle of the unit
fcl = 2.0e6                     #[Pa] compressive strength for the bed joint
phi_l = np.deg2rad(45.0)       # friction angle of the bed joint
omega_max = np.deg2rad(45.0)    # maximum angle related to staircase failure mechanism


G_base = np.array([[t, fcm, ftx, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max]])
G = np.tile(G_base, (number_elements, 1))

# --------------------------------------------------
# Supports
# Left boundary: symmetry => Ux = 0
# Right boundary: shear support => Uy = 0 only on supported segment
#
# IMPORTANT:
# y0 is the free top part on the support boundary.
# I cannot read its exact value from your screenshot, so keep it as a parameter.
# --------------------------------------------------
tol = 1e-10
y0 = 0.25   # <-- set this to the benchmark value if you have it

supports = []

for i, (x, y) in enumerate(node_coordinates):

    # symmetry boundary at x = 0
    if abs(x - 0.0) < tol:
        supports.append([i, 0])   # Ux = 0

    # shear support on right boundary, only for 0 <= y <= h - y0
    if abs(x - lenght) < tol and y <= (height - y0 + tol):
        supports.append([i, 1])   # Uy = 0

supports = np.array(supports, dtype=int)
print("Supports:\n", supports)

# --------------------------------------------------
# Loads
# Uniform vertical load q on the top boundary only
# Use trapezoidal nodal distribution
# --------------------------------------------------
q = 1.0   # reference load intensity [N/m]
tol = 1e-10
loads = []

top_nodes = [i for i, (x, y) in enumerate(node_coordinates)
             if abs(y - height) < tol]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

top_spacing = lenght / (len(top_nodes) - 1)

for k, i in enumerate(top_nodes):
    weight = 0.5 if (k == 0 or k == len(top_nodes)-1) else 1.0
    loads.append([i, 1, -q * top_spacing * weight])

loads = np.array(loads, dtype=float)
print("Loads:\n", loads)

print("Total applied top load =", np.sum(loads[:, 2]))

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
p_collapse = lambda_val * q
print("Collapse load intensity p* =", p_collapse/ (t* 1e6), "MPa")
print("Collapse load intensity p* =", p_collapse/1000, "kN/m")
print("Collapse load intensity p* =", 2 * p_collapse * lenght/1000, "kN")

# --------------------------------------------------
# Post-processing
# --------------------------------------------------
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-6)

from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes, 1e-2)