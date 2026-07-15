import numpy as np


# -----------------------------------------------------------------------------
# Geometry and mesh
# -----------------------------------------------------------------------------
length = 2.0
height = 2.0

nx = 12
ny = 12

a = (length / nx) / 2.0
b = (height / ny) / 2.0

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
    [2*a, 2*b],
])

base_elements = np.array([
    [1, 3, 7, 5, 4, 2],
    [1, 7, 11, 9, 6, 4],
    [3, 13, 7, 10, 5, 8],
    [7, 13, 11, 12, 9, 10],
]) - 1

node_dict = {}
node_list = []
elements = []

for i in range(nx):
    for j in range(ny):
        shift = np.array([2.0*a*i, 2.0*b*j])
        local_nodes = base_nodes + shift
        local_to_global = []

        for node in local_nodes:
            key = tuple(np.round(node, 10))
            if key not in node_dict:
                node_dict[key] = len(node_list)
                node_list.append(node)
            local_to_global.append(node_dict[key])

        local_to_global = np.array(local_to_global, dtype=int)

        for el in base_elements:
            elements.append(local_to_global[el])

node_coordinates = np.array(node_list, dtype=float)
elements_topology = np.array(elements, dtype=int)

number_nodes = node_coordinates.shape[0]
number_elements = elements_topology.shape[0]

print("Number of nodes:", number_nodes)
print("Number of elements:", number_elements)


# -----------------------------------------------------------------------------
# Material parameters
# -----------------------------------------------------------------------------
t = 0.228                       # wall thickness [m]
fcm = 10.57e6                   # effective masonry compressive strength [Pa]
ftx = 0.24e6                    # tensile strength parallel to bed joints [Pa]
nu = 0.7 - 10.57 / 200          # effectiveness factor [-]
fcs = 30e6                      # unit compressive strength [Pa]
fts = ftx                       # unit tensile strength [Pa]
fce = 3.29e6                    # head-joint compressive strength [Pa]
xi = 0.5                        # failure-line area ratio [-]
phi_s = np.deg2rad(30.0)        # friction angle of the bed joint [rad]
fcl = fce                       # bed-joint compressive strength [Pa]
phi_l = np.deg2rad(30.0)        # friction angle of the bed joint [rad]
omega_max = np.deg2rad(45.0)    # maximum angle of staircase failure line [rad]

G_base = np.array([[t, fcm, ftx, nu, fcs, fts, fce, xi, phi_s, fcl, phi_l, omega_max]])
G = np.tile(G_base, (number_elements, 1))


# -----------------------------------------------------------------------------
# Supports
# -----------------------------------------------------------------------------
tol = 1e-10
supports = []

for node, (x, y) in enumerate(node_coordinates):
    if abs(y) < tol:
        supports.append([node, 0])
        supports.append([node, 1])

supports = np.array(supports, dtype=int)



# -----------------------------------------------------------------------------
# Loads
# -----------------------------------------------------------------------------
Fv_total = 105.0e3   # constant vertical top load [N]
Fh_ref_total = 1.0   # reference horizontal load [N]
rho = 1800.0         # masonry density [kg/m^3]
g = 9.81             # gravitational acceleration [m/s^2]

loads_const = []
loads_var = []

# Top nodes over the full wall length
top_nodes = [
    i for i, (x, y) in enumerate(node_coordinates)
    if abs(y - height) < tol
]
top_nodes = sorted(top_nodes, key=lambda i: node_coordinates[i, 0])

# Distribute both loads using tributary widths along the top edge
top_x = node_coordinates[top_nodes, 0]
tributary_width = np.zeros(len(top_nodes))

for k in range(len(top_nodes)):
    if k == 0:
        tributary_width[k] = 0.5 * (top_x[k + 1] - top_x[k])
    elif k == len(top_nodes) - 1:
        tributary_width[k] = 0.5 * (top_x[k] - top_x[k - 1])
    else:
        tributary_width[k] = 0.5 * (top_x[k + 1] - top_x[k - 1])

weights = tributary_width / np.sum(tributary_width)

for k, node in enumerate(top_nodes):
    loads_const.append([node, 1, -Fv_total * weights[k]])
    loads_var.append([node, 0, Fh_ref_total * weights[k]])

loads_const = np.array(loads_const, dtype=float)
loads_var = np.array(loads_var, dtype=float)

# Add self-weight as a permanent load.
from model.loads import add_self_loads
loads_const = add_self_loads(loads_const,node_coordinates,elements_topology,thickness=t,density=rho,g=g,)

# -----------------------------------------------------------------------------
# FELA problem
# -----------------------------------------------------------------------------
from fem.assembly import setup_global_mapping
from fem.equilibrium import global_equilibrium_matrix
from fem.constrains_masonry import setcon_masonry
from model.supports import setsup
from model.loads import setload
from optimization.mosek_solver import solveopt

(number_nodes,number_elements,number_variables,number_equations,variables_per_element,equations_per_element,) = setup_global_mapping(node_coordinates, elements_topology)

global_equilibrium = global_equilibrium_matrix(node_coordinates, elements_topology, number_elements, number_equations, number_variables, equations_per_element, variables_per_element,)

number_sup, support_dofs, global_equilibrium_reduced = setsup(supports,global_equilibrium,)

number_load_const, global_load_vector_const, global_load_vector_reduced_const = setload(2 * number_nodes, support_dofs, loads_const)
number_load_var, global_load_vector_var, global_load_vector_reduced_var = setload(2 * number_nodes, support_dofs, loads_var)


Ab, blc, buc, C = setcon_masonry(number_elements, G)

from optimization.mosek_solver import solveopt
x, alpha, y, lambda_val = solveopt(number_elements,global_equilibrium_reduced,global_load_vector_reduced_const,global_load_vector_reduced_var,Ab, blc, buc, C)


# -----------------------------------------------------------------------------
# Support reactions
# -----------------------------------------------------------------------------
S = x[:9 * number_elements]

global_load_total = global_load_vector_const + lambda_val * global_load_vector_var

residual = global_equilibrium @ S - global_load_total

support_reactions = residual[support_dofs]

print("\nSupport reactions:")
for dof, reaction in zip(support_dofs, support_reactions):
    node = dof // 2
    direction = "x" if dof % 2 == 0 else "y"
    print(
        f"Node {node:4d}, {direction}-reaction = {reaction / 1000:.3f} kN"
    )

print("\nSum of reactions:")
Rx = np.sum(residual[support_dofs[support_dofs % 2 == 0]])
Ry = np.sum(residual[support_dofs[support_dofs % 2 == 1]])

print(f"Sum Rx = {Rx / 1000:.3f} kN")
print(f"Sum Ry = {Ry / 1000:.3f} kN")

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
sx_all = []
sy_all = []
tau_all = []

for el in range(number_elements):
    for gp in range(3):
        idx = 9 * el + 3 * gp
        sx_all.append(x[idx])
        sy_all.append(x[idx + 1])
        tau_all.append(x[idx + 2])

print("lambda =", lambda_val)
print("min sigma_y =", np.min(sy_all))
print("max sigma_y =", np.max(sy_all))
print("Top vertical load [kN] =", Fv_total / 1000.0)
print("Self-weight [kN] =", rho * g * t * length * height / 1000.0)
print("Sum vertical load [kN] =", np.abs(np.sum(loads_const[:, 2]) / 1000.0))
print("Sum horizontal collapse load [kN] =", (np.sum(loads_var[:, 2])*lambda_val)/1000.0)

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
from post.plot_topology import plot_topology
from post.plot_principle_stress import plotPS_contour
from post.plot_displacements import plotDof

plot_topology(node_coordinates, elements_topology)
plotPS_contour(node_coordinates, elements_topology, x, 1e-6,)
plotDof(node_coordinates, elements_topology, y, support_dofs, 1e-1)
