import numpy as np

def setsup(S, H):
    nsup = S.shape[0]
    supeq = 2*S[:, 0] + (S[:, 1] - 1)   # node is 0-based, dir is 1/2
    supeq = supeq.astype(int)
    Hsup = np.delete(H, supeq, axis=0)
    return nsup, supeq, Hsup