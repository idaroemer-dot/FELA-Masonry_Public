from scipy.sparse import lil_matrix
import numpy as np

def Ab_RC_ps(fc, phix, phiy):
    Aseq = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=float)

    Aaeq = np.array([
        [0,  -1,   0,  0, 0, 0, -1,  0],
        [0,   0,  -1,  0, 0, 0,  0, -1],
        [0,   0,   0, -1, 0, 0,  0,  0],
        [0, -0.5, 0.5, 0, 1, 0,  0,  0],
        [0, -0.5,-0.5, 0, 0, 1,  0,  0],
    ], dtype=float)

    byeq = np.zeros(5, dtype=float)

    As = np.zeros((6, 3), dtype=float)

    Aa = np.array([
        [0, 0, 0, 0, 0,  0, -1,  0],
        [0, 0, 0, 0, 0,  0,  1,  0],
        [0, 0, 0, 0, 0,  0,  0, -1],
        [0, 0, 0, 0, 0,  0,  0,  1],
        [1, 0, 0, 0, 0,  1,  0,  0],
        [1, 0, 0, 0, 0, -1,  0,  0],
    ], dtype=float)

    by = np.array([0, phix*fc, 0, phiy*fc, 0, fc], dtype=float)

    return Aseq, Aaeq, byeq, As, Aa, by


def setcon(nel, G):
    na = 8
    nr = 11

    Ab  = lil_matrix((3*nel*nr, 9*nel + 1 + 3*nel*na))
    blc = np.zeros(3*nel*nr, dtype=float)
    buc = np.zeros(3*nel*nr, dtype=float)
    C   = [None] * (3*nel)

    for el in range(nel):
        for no in range(3):
            t  = G[el, 0]
            fc = G[el, 1]
            phix = G[el, 2] * G[el, 3] / (G[el, 0] * G[el, 1])
            phiy = G[el, 4] * G[el, 5] / (G[el, 0] * G[el, 1])

            Aseq, Aaeq, byeq, As, Aa, by = Ab_RC_ps(fc, phix, phiy)

            rp = (3*el + no) * nr
            r  = slice(rp, rp + nr)

            cp_beta = 9*el + no*3
            Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))

            cp_alfa = 9*nel + 1 + (3*el + no) * na
            Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

            blc[r] = np.concatenate((byeq, -np.inf*np.ones_like(by)))
            buc[r] = np.concatenate((byeq, by))

            C[3*el + no] = {
                "type": "MSK_CT_QUAD",
                "sub": cp_alfa + np.array([0, 3, 4])  # MATLAB [1 4 5] -> Python [0 3 4]
            }

    return Ab.tocsr(), blc, buc, C
