from scipy.sparse import lil_matrix
import numpy as np

def Ab_steel_vm_ps(fy):
    """
    Steel: von Mises in 2D plane stress
    Compatible with the mixed material framework.

    Global convention:
        na = 10
        nr = 14

    beta = [sigma_x, sigma_y, tau_xy]

    alpha = [
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
    ]

    Only a0..a3 are used physically.
    a4..a9 are dummy padding variables.

    The von Mises reformulation is

        a0 = fy
        a1 = -sqrt(3)/2 * sigma_x + sqrt(3)/2 * sigma_y
        a2 = -1/2 * sigma_x - 1/2 * sigma_y
        a3 = -sqrt(3) * tau_xy

    with cone:
        (a0, a1, a2, a3) in Q4
    """

    # ---------------------------------
    # Equalities: 4 rows
    # Aseq * beta + Aaeq * alpha = byeq
    # ---------------------------------
    Aseq = np.array([
        [ 0.0,              0.0,             0.0],
        [-np.sqrt(3)/2.0,   np.sqrt(3)/2.0,  0.0],
        [-0.5,             -0.5,             0.0],
        [ 0.0,              0.0,            -np.sqrt(3)],
    ], dtype=float)

    Aaeq = np.array([
        [-1.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -a0 = -fy
        [ 0.0, -1.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -a1
        [ 0.0,  0.0, -1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -a2
        [ 0.0,  0.0,  0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -a3
    ], dtype=float)

    byeq = np.array([
        -fy,
         0.0,
         0.0,
         0.0,
    ], dtype=float)

    # ---------------------------------
    # No linear inequalities for pure von Mises
    # Pad with 10 dummy rows so nr = 4 + 10 = 14
    # ---------------------------------
    As = np.zeros((10, 3), dtype=float)
    Aa = np.zeros((10, 10), dtype=float)
    by = np.zeros(10, dtype=float)

    return Aseq, Aaeq, byeq, As, Aa, by


def setcon_steel_vm(nel, G):
    """
    G[el, 0] = thickness t
    G[el, 1] = fy

    Compatible with the same mixed-format assembly idea as the
    masonry and RC blocks.
    """

    na = 10
    neq = 4
    nin = 10
    nr = neq + nin   # = 14

    Ab  = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr, dtype=float)
    buc = np.zeros(3 * nel * nr, dtype=float)

    # one cone per constitutive point
    C = [None] * (3 * nel)

    for el in range(nel):
        for no in range(3):
            t  = G[el, 0]
            fy = G[el, 1]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_steel_vm_ps(fy)

            rp = (3 * el + no) * nr
            r  = slice(rp, rp + nr)

            # beta block
            cp_beta = 9 * el + no * 3
            Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))

            # alpha block
            cp_alfa = 9 * nel + 1 + (3 * el + no) * na
            Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

            # equalities + padded inequalities
            blc[r] = np.concatenate((byeq, -np.inf * np.ones_like(by)))
            buc[r] = np.concatenate((byeq, by))

            # von Mises cone: a0 >= sqrt(a1^2 + a2^2 + a3^2)
            C[3 * el + no] = {
                "type": "MSK_CT_QUAD",
                "sub": cp_alfa + np.array([0, 1, 2, 3])
            }

    return Ab.tocsr(), blc, buc, C