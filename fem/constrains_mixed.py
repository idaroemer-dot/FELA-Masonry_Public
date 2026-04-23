from scipy.sparse import lil_matrix
import numpy as np

from fem.constrains_masonry import Ab_masonry_ps
from fem.constrains_RC import Ab_RC_ps


def setcon_mixed(nel, G, mat_type):
    """
    Mixed constitutive assembler.

    Element types:
        0 = masonry web
        1 = masonry flange
        2 = RC beam

    Unified block sizes:
        na = 10
        nr = 14

    Cones are appended dynamically:
        masonry -> 3 cones / GP
        RC      -> 1 cone  / GP
    """
    na = 10
    nr = 14

    Ab  = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr, dtype=float)
    buc = np.zeros(3 * nel * nr, dtype=float)
    C   = []

    for el in range(nel):
        for no in range(3):
            rp = (3 * el + no) * nr
            r  = slice(rp, rp + nr)

            cp_beta = 9 * el + no * 3
            cp_alfa = 9 * nel + 1 + (3 * el + no) * na

            t = G[el, 0]

            # ---------------------------------
            # RC beam element
            # ---------------------------------
            if mat_type[el] == 2:
                fc   = G[el, 1]
                phix = G[el, 2]
                phiy = G[el, 3]

                Aseq, Aaeq, byeq, As, Aa, by = Ab_RC_ps(fc, phix, phiy)

                Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))
                Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

                blc[r] = np.concatenate((byeq, -np.inf * np.ones_like(by)))
                buc[r] = np.concatenate((byeq, by))

                C.append({
                    "type": "MSK_CT_QUAD",
                    "sub": cp_alfa + np.array([0, 3, 4])
                })

            # ---------------------------------
            # Masonry element
            # ---------------------------------
            else:
                fcm       = G[el, 1]
                ftx       = G[el, 2]
                nu        = G[el, 3]
                fcs       = G[el, 4]
                fce       = G[el, 5]
                xi        = G[el, 6]
                phi_s     = G[el, 7]
                fcl       = G[el, 8]
                phi_l     = G[el, 9]
                omega_max = G[el, 10]

                Aseq, Aaeq, byeq, As, Aa, by = Ab_masonry_ps(
                    ftx, fcm, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max
                )

                Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))
                Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

                blc[r] = np.concatenate((byeq, -np.inf * np.ones_like(by)))
                buc[r] = np.concatenate((byeq, by))

                # Cone 1: a1 >= sqrt(a2^2 + a3^2)
                C.append({
                    "type": "MSK_CT_QUAD",
                    "sub": cp_alfa + np.array([0, 1, 2])
                })

                # Cone 2: 2*a5*a6 >= a7^2
                C.append({
                    "type": "MSK_CT_RQUAD",
                    "sub": cp_alfa + np.array([4, 5, 6])
                })

                # Cone 3: 2*a8*a9 >= a10^2
                C.append({
                    "type": "MSK_CT_RQUAD",
                    "sub": cp_alfa + np.array([7, 8, 9])
                })

    return Ab.tocsr(), blc, buc, C