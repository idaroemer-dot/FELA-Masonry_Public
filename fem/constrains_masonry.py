from scipy.sparse import lil_matrix
import numpy as np

def Ab_masonry_ps(ftx, fcm, nu):
    Aseq = np.array([
        [ 0,     0, -1],
        [-0.5,  0.5, 0],
        [-0.5, -0.5, 0],
    ], dtype=float)

    Aaeq = np.array([
        [0,   1,   0,  0],
        [0,   0,   1,  0],
        [0,   0,   0,  1],
    ], dtype=float)

    byeq = np.zeros(3)

    Aaeq_extra = np.array([
    # z1 - 2*alpha2 = 0
    [0, -2, 0, 0, 0, 1, 0],

    # z2 - 2*alpha3 - 2*alpha4 = B
    [0, 0, -2, -2, 0, 0, 1],

    # z0 = A
    [0, 0, 0, 0, 1, 0, 0],
    ])

    by = np.array([
        ftx,
        fcm,
        0.0,
        0.5*nu*fcm,
        0.5*nu*fcm,
    ], dtype=float)

    byeq_extra = np.array([
    0.0,
    B,
    A
    ])

    Aaeq = np.vstack((Aaeq, Aaeq_extra))
    byeq = np.concatenate((byeq, byeq_extra))

    As = np.zeros((5, 3), dtype=float)

    # --- masonry limits --- #
    Aa = np.array([
        [1.0, 0.0,  0.0,  1.0],   # Ia
        [1.0, 0.0,  0.0, -1.0],   # II
        [0.0, 0.0, -1.0,  1.0],   # Ib
        [0.0,  1.0, 0.0, 0.0],    # VIIIa
        [0.0, -1.0, 0.0, 0.0],    # VIIIb
    ], dtype=float)



    return Aseq, Aaeq, byeq, As, Aa, by


def setcon_masonry(nel, G):
    na = 7
    neq = 6
    nin = 5
    nr = neq+nin

    Ab  = lil_matrix((3*nel*nr, 9*nel + 1 + 3*nel*na))
    blc = np.zeros(3*nel*nr, dtype=float)
    buc = np.zeros(3*nel*nr, dtype=float)
    C   = [None] * (3*nel)

    for el in range(nel):
        for no in range(3):
            t  = G[el, 0]
            fcm = G[el, 1]
            ftx = G[el, 2]
            nu = G[el, 3]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_masonry_ps(ftx, fcm, nu)

            rp = (3*el + no) * nr
            r  = slice(rp, rp + nr)

            cp_beta = 9*el + no*3
            Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))

            cp_alfa = 9*nel + 1 + (3*el + no) * na
            Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

            blc[r] = np.concatenate((byeq, -np.inf*np.ones_like(by))) # bound vaulues for lower constraints
            buc[r] = np.concatenate((byeq, by))                       # bound values for upper constraints

            C[3*el + no] = [
                {
                    "type": "MSK_CT_QUAD",
                    "sub": cp_alfa + np.array([0, 1, 2])  # gammel cone
                },
                {
                    "type": "MSK_CT_QUAD",
                    "sub": cp_alfa + np.array([4, 5, 6])  # ny cone (z0,z1,z2)
                }
            ]
    return Ab.tocsr(), blc, buc, C