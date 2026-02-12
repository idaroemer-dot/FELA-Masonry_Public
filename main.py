from matplotlib import scale
import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista
import matplotlib.tri as mtri


#Topology 
#define nodes
a = 0.1
# node_coordinates = [x, y]
node_coordinates = np.array([
    [0, 0],
    [a, 0],
    [2*a, 0],
    [0.5*a, 0.5*a],
    [1.5*a, 0.5*a],
    [0, a],
    [a, a],
    [2*a, a],
    [0.5*a, 1.5*a],
    [1.5*a, 1.5*a],
    [0, 2*a],
    [a, 2*a],
    [2*a, 2*a]
])

# elements_topology = [nodes counter clockwise]
elements_topology = np.array([
    [1, 3, 7, 5, 4, 2],
    [1, 7, 11, 9, 6, 4],
    [3, 13, 7, 10, 5, 8],
    [7, 13, 11, 12, 9, 10]
]) - 1  # convert to zero-based indexing

# materials: G[el] = [t, fc, Ax, fYx, Ay, fYy]
t = 0.25
fc = 20e6
Ax = 2 * (np.pi / 4) * 0.01**2 / 0.2
fYx = 430e6
Ay = Ax
fYy = 430e6
G = np.array([
    [t, fc, Ax, fYx, Ay, fYy],
    [t, fc, Ax, fYx, Ay, fYy],
    [t, fc, Ax, fYx, Ay, fYy],
    [t, fc, Ax, fYx, Ay, fYy]
])

# supports[i] = [node, direction]
supports = np.array([
    [0, 0],
    [0, 1],
    [2, 1]
])

# loads[i] = [node, direction, value]
f = 0.2 * 0.25 / 6
loads = np.array([
    [0, 0, -1*f],
    [1, 0, -4*f],
    [2, 0, -1*f],
    [2, 1, 1*f],
    [7, 1, 4*f],
    [12, 1, 1*f],
    [10, 0, 1*f],
    [11, 0, 4*f],
    [12, 0, 1*f],
    [0, 1, -1*f],
    [5, 1, -4*f],
    [10, 1, -1*f]
])

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
from fem.constrains import setcon
Ab, blc, buc, C = setcon(number_elements, G)

#Optimize
from optimization.mosek_solver import solveopt
x, y = solveopt(number_elements, global_equilibrium_reduced, global_load_vector_reduced, Ab, blc, buc, C)

#Plot principal stresses
from post.plot_principle_stress import plotPS
plotPS(node_coordinates, elements_topology, x, number_elements, 1e-7)

#Plot displacements
from post.plot_displacements import plotDof
plotDof(node_coordinates, elements_topology, y, global_DOF_index_supports, number_elements, number_nodes, 1)