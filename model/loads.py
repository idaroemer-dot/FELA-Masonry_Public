import numpy as np

def setload(neq, supeq, L):
    R = np.zeros(neq)
    nload = L.shape[0]

    idx = 2*L[:, 0] + (L[:, 1] - 1)   # node is 0-based, dir is 1/2
    idx = idx.astype(int)

    R[idx] += L[:, 2]

    Rsup = np.delete(R, supeq)

    return nload, R, Rsup
