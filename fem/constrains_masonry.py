from scipy.sparse import lil_matrix
import numpy as np

def Ab_masonry_simple(fcm, ftm, km):

    # beta = [sx, sy, tau]
    # alpha = [u1,v1,w1, u2,v2,w2, u3,v3,w3]

    Aseq = np.array([
        [-1, 0, 0],   # u1 = ftx - sx
        [0, -1, 0],   # v1 = ftx - sy
        [0, 0, 1],    # w1 = tau

        [1, 0, 0],    # u2 = fcm + sx
        [0, 1, 0],    # v2 = fcm + sy
        [0, 0, 1],    # w2 = tau

        [-km, 1, 0],  # u3 = fcm - km*sx + sy
        [1, -km, 0],  # v3 = fcm - km*sy + sx
        [0, 0, 1]     # w3 = tau
    ])

    Aaeq = np.eye(9)

    byeq = np.array([
        ftm,
        ftm,
        0,

        fcm,
        fcm,
        0,

        fcm,
        fcm,
        0
    ])

    As = np.zeros((0,3))
    Aa = np.zeros((0,9))
    by = np.zeros(0)

    C_local = [
        {"type":"MSK_CT_RQUAD","sub":[0,1,2]},
        {"type":"MSK_CT_RQUAD","sub":[3,4,5]},
        {"type":"MSK_CT_RQUAD","sub":[6,7,8]},
    ]

    return Aseq, Aaeq, byeq, C_local

def setcon_masonry(nel, G):

    na = 9     # auxiliary variables
    nr = 9     # constraints per stress point

    Ab  = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr)
    buc = np.zeros(3 * nel * nr)

    C = []

    for el in range(nel):
        for no in range(3):

            t   = G[el,0]
            fcm = G[el,1]
            ftm = G[el,2]
            km  = G[el,3]

            Aseq, Aaeq, byeq, C_local = Ab_masonry_simple(fcm, ftm, km)

            rp = (3 * el + no) * nr
            r  = slice(rp, rp + nr)

            cp_beta = 9 * el + no * 3
            Ab[r, cp_beta:cp_beta+3] = Aseq / t

            cp_alfa = 9 * nel + 1 + (3 * el + no) * na
            Ab[r, cp_alfa:cp_alfa+na] = Aaeq

            blc[r] = byeq
            buc[r] = byeq

            for c in C_local:
                C.append({
                    "type": c["type"],
                    "sub": cp_alfa + np.array(c["sub"])
                })

    return Ab.tocsr(), blc, buc, C