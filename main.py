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
# node coordinates: X[node] = [x, y]
X = np.array([
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

# elements: T[el] = [nodes counter clockwise]
T = np.array([
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

# supports: S[i] = [node, dir]
S = np.array([
    [0, 1],
    [0, 2],
    [2, 2]
])


# loads: L[i] = [node, dir, val]
f = 0.2 * 0.25 / 6
L = np.array([
    [0, 1, -1*f],
    [1, 1, -4*f],
    [2, 1, -1*f],
    [2, 2, 1*f],
    [7, 2, 4*f],
    [12, 2, 1*f],
    [10, 1, 1*f],
    [11, 1, 4*f],
    [12, 1, 1*f],
    [0, 2, -1*f],
    [5, 2, -4*f],
    [10, 2, -1*f]
])

#Setup global mapping
from fem.equilibrium import setup_global_mapping
number_nodes, number_elements, number_variabels, number_equations, variables_per_element, equations_per_element = setup_global_mapping(X, T)

#Plot geometry
from post.plot_topology import plot_topology
plotTop = plot_topology(X, T, number_elements, number_nodes)


#Establish equilibrium matrix

# connectivity helper (MATLAB cy matrix, 1-based → Python 0-based)
cy = np.array([
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1]
])

# ------------------------------------------------------------------
# hplst: compute element equilibrium matrix
# ------------------------------------------------------------------
def hplst(X, T):
    neq, nvar = 12, 9
    h = np.zeros((neq, nvar))

    a = np.zeros(3)
    b = np.zeros(3)

    # geometry
    for i in range(3):
        j = cy[1, i]
        k = cy[2, i]
        a[i] = X[k, 0] - X[j, 0]
        b[i] = X[k, 1] - X[j, 1]

    P = []
    for i in range(3):
        P.append(np.array([
            [ b[i],   0.0, -a[i]],
            [ 0.0, -a[i],  b[i]]
        ]))

    O23 = np.zeros((2, 3))

    h = np.block([
        [-P[0],        O23,        O23],
        [ O23,       -P[1],        O23],
        [ O23,         O23,      -P[2]],
        [ P[0],   P[0]-P[2],  P[0]-P[1]],
        [P[1]-P[2],    P[1],  P[1]-P[0]],
        [P[2]-P[1], P[2]-P[0],     P[2]]
    ]) / 6.0

    return h


# ------------------------------------------------------------------
# seth: establish global equilibrium matrix
# ------------------------------------------------------------------
def seth(X, T, nel, neq, nvar, equations_per_element, variables_per_element):
    H = np.zeros((neq, nvar))

    for el in range(nel):
        h = hplst(X[T[el, :], :], T[el, :])
        H[np.ix_(equations_per_element[el], variables_per_element[el])] = h

    return H

H = seth(X, T, nel, neq, nvar, equations_per_element, variables_per_element)

#Set supports
from model.supports import setsup
nsup, supeq, Hsup = setsup(S, H)

#Set loads 
from model.loads import setload
nload, R, Rsup = setload(2*nno, supeq, L)

#Set constraints
from fem.constrains import setcon
Ab, blc, buc, C = setcon(nel, G)

#Optimize
from optimization.mosek_solver import solveopt
x, y = solveopt(nel, Hsup, Rsup, Ab, blc, buc, C)

#Plot principal stresses
from post.plot_principle_stress import plotPS
plotPS(X, T, x, nel, 1e-7)

#Plot displacements
from post.plot_displacements import plotDof
plotDof(X, T, y, supeq, nel, nno, 1)