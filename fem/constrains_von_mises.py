
from scipy.sparse import lil_matrix
import numpy as np

def Ab_steel_vm_ps(fy):
    """
    Von Mises yield condition in 2D plane stress

    beta = [sx, sy, txy]

    alpha = [a0, a1, a2, a3]

    with
        a0 = fy
        a1 = -sqrt(3)/2 * sx + sqrt(3)/2 * sy
        a2 = -1/2 * sx - 1/2 * sy
        a3 = -sqrt(3) * txy

    and cone:
        (a0, a1, a2, a3) in Q4
    """

    # Equalities: Aseq * beta + Aaeq * alpha = byeq
    Aseq = np.array([
        [0.0,               0.0,              0.0],
        [-np.sqrt(3)/2.0,   np.sqrt(3)/2.0,   0.0],
        [-0.5,             -0.5,              0.0],
        [0.0,               0.0,             -np.sqrt(3)],
    ], dtype=float)

    Aaeq = -np.eye(4, dtype=float)

    byeq = np.array([
        -fy,
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    # No linear inequalities for pure von Mises
    As = np.zeros((0, 3), dtype=float)
    Aa = np.zeros((0, 4), dtype=float)
    by = np.zeros(0, dtype=float)

    return Aseq, Aaeq, byeq, As, Aa, by


def setcon_steel_vm(nel, G):
    """
    G[el, 0] = thickness t
    G[el, 1] = fy
    """

    na = 4   # number of auxiliary variables per constitutive point
    nr = 4   # number of local coupling equations per constitutive point

    Ab  = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr, dtype=float)
    buc = np.zeros(3 * nel * nr, dtype=float)
    C   = [None] * (3 * nel)

    for el in range(nel):
        for no in range(3):
            t  = G[el, 0]
            fy = G[el, 1]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_steel_vm_ps(fy)

            rp = (3 * el + no) * nr
            r  = slice(rp, rp + nr)

            # stress variables for this constitutive point
            cp_beta = 9 * el + no * 3
            Ab[r, slice(cp_beta, cp_beta + 3)] = Aseq / t

            # auxiliary variables for this constitutive point
            cp_alfa = 9 * nel + 1 + (3 * el + no) * na
            Ab[r, slice(cp_alfa, cp_alfa + na)] = Aaeq

            blc[r] = byeq
            buc[r] = byeq

            # cone: alpha[0] >= sqrt(alpha[1]^2 + alpha[2]^2 + alpha[3]^2)
            C[3 * el + no] = {
                "type": "MSK_CT_QUAD",
                "sub": cp_alfa + np.array([0, 1, 2, 3])
            }

    return Ab.tocsr(), blc, buc, C