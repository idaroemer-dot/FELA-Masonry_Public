import numpy as np

# ----- base cell -----

a = 0.1
b = 0.05

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


# ----- number of macro cells -----

nx = 10
ny = 10

all_nodes = []
all_elements = []

node_offset = 0

for i in range(nx):
    for j in range(ny):

        # shift cell
        shift = np.array([2*a*i, 2*b*j])
        new_nodes = base_nodes + shift

        all_nodes.append(new_nodes)

        # shift element connectivity
        new_elements = base_elements + node_offset
        all_elements.append(new_elements)

        node_offset += base_nodes.shape[0]


node_coordinates = np.vstack(all_nodes)
elements_topology = np.vstack(all_elements)

number_nodes = node_coordinates.shape[0]
number_elements = elements_topology.shape[0]

import matplotlib.pyplot as plt


# plot geometry
plt.figure(1)
plt.title('Element topologi')
plt.axis('equal')
local_dof = [0, 5, 1, 3, 2, 4, 0] # local dof indices for plotting element edges

for el in range(number_elements):
    plt.plot(node_coordinates[elements_topology[el, local_dof], 0], node_coordinates[elements_topology[el, local_dof], 1], 'b-')

for el in range(number_elements): # write element numbers
    xp = np.mean(node_coordinates[elements_topology[el, :], :], axis=0)
    plt.text(xp[0], xp[1], str(el + 1), color='black')

plt.show()

#cross
lenght = 1           # Total length   
height = 1    # Total height

nx = 17
ny = 17

a = (lenght/nx)/2     # Length of cell in x direction
b = (height/ny)/2     # Height of cell in y direction

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


# multiple cells
all_nodes = []
all_elements = []

node_dict = {}        # maps coordinate tuple -> global index
node_list = []        # final unique nodes

for i in range(nx):
    for j in range(ny):

        shift = np.array([2*a*i, 2*b*j])
        shifted_nodes = base_nodes + shift

        local_to_global = []

        # --- build global node numbering ---
        for node in shifted_nodes:
            key = tuple(np.round(node, 10))  # avoid float issues

            if key in node_dict:
                global_index = node_dict[key]
            else:
                global_index = len(node_list)
                node_dict[key] = global_index
                node_list.append(node)

            local_to_global.append(global_index)

        local_to_global = np.array(local_to_global)

        # --- build elements using mapped indices ---
        for el in base_elements:
            all_elements.append(local_to_global[el])

#diagonal
hol = 0.5
l = 1.0
h = hol * l * 2
n = 16

# Nodes
dx = l / (2*n)
dy = h / (2*n)

X = []

for iy in range(2*n + 1):
    for ix in range(2*n + 1):
        X.append([(ix) * dx, (iy) * dy])

X = np.array(X)
nno = X.shape[0]

# Elements 

T = []

# First element (convert MATLAB → 0-based indexing)
base = np.array([
    1,
    3,
    1 + (2*n+1)*2,
    2 + (2*n+1),
    1 + (2*n+1),
    2
]) - 1

T.append(base)

# shift in x-direction
dTx = np.array([2,2,2,2,2,2])

for ix in range(1, n):
    T.append(T[-1] + dTx)

# middle transition row
mid = np.array([
    1 + (2*n+1)*2,
    3,
    3 + (2*n+1)*2,
    3 + (2*n+1),
    2 + (2*n+1)*2,
    2
]) - 1

T.append(mid)

for ix in range(1, n):
    T.append(T[-1] + dTx)

T = np.array(T)

# copy first block
T1 = T.copy()

# vertical shift
dTy = (2*n+1)*2 * np.ones((2*n,6), dtype=int)

for iy in range(1, n):
    T = np.vstack((T, T1 + dTy*iy))

nel = T.shape[0]

print("Number of nodes:", nno)
print("Number of elements:", nel)