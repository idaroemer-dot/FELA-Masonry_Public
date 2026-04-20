from turtle import left

from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix


def apply_rigid_top_ux(global_equilibrium_reduced,
                       global_load_vector_reduced_const,
                       global_load_vector_reduced_var,
                       number_nodes,
                       global_DOF_index_supports,
                       top_nodes):
    """
    Enforce ux(top_i) = ux(top_master) for all top nodes
    by merging the corresponding reduced equilibrium rows
    and reduced load entries.
    """

    supported_dofs = np.array(global_DOF_index_supports, dtype=int).ravel()
    all_dofs = np.arange(2 * number_nodes, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, supported_dofs)

    top_ux_dofs = [2 * i for i in top_nodes]
    dof_to_reduced_row = {dof: r for r, dof in enumerate(free_dofs)}
    active_rows = [dof_to_reduced_row[d] for d in top_ux_dofs if d in dof_to_reduced_row]

    if len(active_rows) <= 1:
        tie_data = {
            "keep_mask": np.ones(global_equilibrium_reduced.shape[0], dtype=bool),
            "master_row_before": None,
            "slave_rows_before": [],
        }
        return (global_equilibrium_reduced,
                global_load_vector_reduced_const,
                global_load_vector_reduced_var,
                tie_data)

    master_row = active_rows[0]
    slave_rows = active_rows[1:]

    H = lil_matrix(global_equilibrium_reduced.copy())
    R0 = np.array(global_load_vector_reduced_const, dtype=float).reshape(-1).copy()
    R = np.array(global_load_vector_reduced_var, dtype=float).reshape(-1).copy()

    for r in slave_rows:
        H[master_row, :] = H.getrow(master_row) + H.getrow(r)
        R0[master_row] += R0[r]
        R[master_row] += R[r]

    keep_mask = np.ones(H.shape[0], dtype=bool)
    keep_mask[slave_rows] = False

    H_tied = H[keep_mask, :].tocsr()
    R0_tied = R0[keep_mask]
    R_tied = R[keep_mask]

    tie_data = {
        "keep_mask": keep_mask,
        "master_row_before": master_row,
        "slave_rows_before": slave_rows,
    }

    return H_tied, R0_tied, R_tied, tie_data


def recover_tied_displacements(y_tied, global_equilibrium_reduced, tie_data):
    """
    Recover displacement vector with the original reduced size
    (before tying rows), so existing plotDof still works.
    """
    y_tied = np.array(y_tied, dtype=float).reshape(-1)

    n_red = global_equilibrium_reduced.shape[0]
    y_red = np.zeros(n_red)

    keep_mask = tie_data["keep_mask"]
    y_red[keep_mask] = y_tied

    master_row = tie_data["master_row_before"]
    slave_rows = tie_data["slave_rows_before"]

    if master_row is not None:
        for r in slave_rows:
            y_red[r] = y_red[master_row]

    return y_red


#---------Topology---------#
lenght = 0.99
height = 1.00

nx = 8
ny = 8

a = (lenght / nx) / 2
b = (height / ny) / 2

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

cell_w = 2 * a
cell_h = 2 * b

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

        shift = np.array([2 * a * i, 2 * b * j])
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
# masonry case I
t = 0.100
nu = 1
nu_t = 0.6
nu_c = 0.7 - 10.5 / 200
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

G_base = np.array([[t, fcm, ftx, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max]])
G = np.tile(G_base, (number_elements, 1))

# supports[i] = [node, direction]
tol = 1e-10
supports = []

for i, (x, y) in enumerate(node_coordinates):
    if abs(y - 0.0) < tol:
        supports.append([i, 0])   # Ux = 0
        supports.append([i, 1])   # Uy = 0

supports = np.array(supports, dtype=int)

print(supports)

# loads[i] = [node, direction, value]
Fv_total = 105.0e3   # [N]
Fh_ref_total = 1.0   # [N]

loads_const = []
loads_var = []

top_nodes = [i for i, (x, y) in enumerate(node_coordinates) if abs(y - height) < tol]
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

print(loads_const)
print(loads_var)
print("Sum vertical load [kN] =", np.sum(loads_const[:, 2]) / 1000)
print("Sum horizontal reference load [N] =", np.sum(loads_var[:, 2]))

#Setup global mapping
from fem.assembly import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(
    node_coordinates, elements_topology
)

#Plot geometry
from post.plot_topology import plot_topology
plotTop = plot_topology(node_coordinates, elements_topology, number_elements, number_nodes)

#Establish equilibrium matrix
from fem.equilibrium import global_equilibrium_matrix
global_equilibrium = global_equilibrium_matrix(
    node_coordinates, elements_topology, number_elements, number_equations,
    number_variabels, equations_per_element, variables_per_element
)

#Set supports
from model.supports import setsup
number_sup, global_DOF_index_supports, global_equilibrium_reduced = setsup(supports, global_equilibrium)

#Set loads
from model.loads import setload
number_load_const, global_load_vector_const, global_load_vector_reduced_const = setload(
    2 * number_nodes, global_DOF_index_supports, loads_const
)
number_load_var, global_load_vector_var, global_load_vector_reduced_var = setload(
    2 * number_nodes, global_DOF_index_supports, loads_var
)

# Apply rigid horizontal motion at the top: ux_i = ux_master
global_equilibrium_reduced_tied, global_load_vector_reduced_const_tied, global_load_vector_reduced_var_tied, tie_data = apply_rigid_top_ux(
    global_equilibrium_reduced,
    global_load_vector_reduced_const,
    global_load_vector_reduced_var,
    number_nodes,
    global_DOF_index_supports,
    top_nodes
)

#Set constraints
from fem.constrains_masonry import setcon_masonry
Ab, blc, buc, C = setcon_masonry(number_elements, G)

#Optimize
from optimization.mosek_solver import solveopt
x, alpha, y_tied, lambda_val = solveopt(
    number_elements,
    global_equilibrium_reduced_tied,
    global_load_vector_reduced_const_tied,
    global_load_vector_reduced_var_tied,
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

# collapse load
Fh_collapse = lambda_val * Fh_ref_total
print("Vertical load =", -np.sum(loads_const[:, 2]) / 1000, "kN")
print("Total horizontal collapse load =", Fh_collapse / 1000, "kN")

sigma_eq = Fh_collapse / (t * height)
print("Equivalent horizontal stress =", sigma_eq / 1e6, "MPa")

#Plot principal stresses
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-6)

# Recover displacements before plotting
y = recover_tied_displacements(y_tied, global_equilibrium_reduced, tie_data)

#Plot displacements
from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes, 1e-2)