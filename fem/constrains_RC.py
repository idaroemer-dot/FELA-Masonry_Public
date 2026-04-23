from scipy.sparse import lil_matrix
import numpy as np

def Ab_RC_ps(fc, phix, phiy):
    """
    RC local formulation rewritten to be compatible with the masonry block sizes.

    Global convention for mixed formulation:
        na = 10
        nr = 14

    beta = [sigma_x, sigma_y, tau_xy]

    alpha = [
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10
    ]

    Only a1..a8 are used by the original RC formulation.
    a9 and a10 are padded dummy variables.
    """

    # -----------------------------
    # Original RC equalities: 5 rows
    # -----------------------------
    Aseq = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=float)

    # Original RC alpha coupling was (5 x 8), now padded to (5 x 10)
    Aaeq = np.array([
        [0,  -1,   0,  0, 0, 0, -1,  0, 0, 0],
        [0,   0,  -1,  0, 0, 0,  0, -1, 0, 0],
        [0,   0,   0, -1, 0, 0,  0,  0, 0, 0],
        [0, -0.5, 0.5, 0, 1, 0,  0,  0, 0, 0],
        [0, -0.5,-0.5, 0, 0, 1,  0,  0, 0, 0],
    ], dtype=float)

    byeq = np.zeros(5, dtype=float)

    # -----------------------------
    # Original RC inequalities: 6 rows
    # Pad to 9 rows so total nr = 5 + 9 = 14
    # -----------------------------
    As = np.zeros((9, 3), dtype=float)

    Aa = np.array([
        [0, 0, 0, 0, 0,  0, -1,  0, 0, 0],
        [0, 0, 0, 0, 0,  0,  1,  0, 0, 0],
        [0, 0, 0, 0, 0,  0,  0, -1, 0, 0],
        [0, 0, 0, 0, 0,  0,  0,  1, 0, 0],
        [1, 0, 0, 0, 0,  1,  0,  0, 0, 0],
        [1, 0, 0, 0, 0, -1,  0,  0, 0, 0],
        [0, 0, 0, 0, 0,  0,  0,  0, 0, 0],  # dummy
        [0, 0, 0, 0, 0,  0,  0,  0, 0, 0],  # dummy
        [0, 0, 0, 0, 0,  0,  0,  0, 0, 0],  # dummy
    ], dtype=float)

    by = np.array([
        0,
        phix * fc,
        0,
        phiy * fc,
        0,
        fc,
        0,   # dummy
        0,   # dummy
        0,   # dummy
    ], dtype=float)

    return Aseq, Aaeq, byeq, As, Aa, by


def setcon(nel, G):
    """
    Standalone RC assembler, now also using:
        na = 10
        nr = 14
    """
    na = 10
    nr = 14

    Ab  = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr, dtype=float)
    buc = np.zeros(3 * nel * nr, dtype=float)
    C   = []

    for el in range(nel):
        for no in range(3):
            t  = G[el, 0]
            fc = G[el, 1]
            phix = G[el, 2]
            phiy = G[el, 3]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_RC_ps(fc, phix, phiy)

            rp = (3 * el + no) * nr
            r  = slice(rp, rp + nr)

            cp_beta = 9 * el + no * 3
            Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))

            cp_alfa = 9 * nel + 1 + (3 * el + no) * na
            Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

            blc[r] = np.concatenate((byeq, -np.inf * np.ones_like(by)))
            buc[r] = np.concatenate((byeq, by))

            C.append({
                "type": "MSK_CT_QUAD",
                "sub": cp_alfa + np.array([0, 3, 4])
            })

    return Ab.tocsr(), blc, buc, C