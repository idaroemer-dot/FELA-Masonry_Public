from scipy.sparse import lil_matrix
import numpy as np

def Ab_masonry(ctau, cn, mu, hb, lb, f):

    rgeom = 2*hb/lb
    sgeom = lb/(2*hb)
    kgeom = f*lb/(2*hb)

    # equality constraints
    Aseq = np.zeros((0,3))
    Aaeq = np.zeros((0,0))
    byeq = np.zeros(0)

    #second order cone constraints
    As = np.zeros((0,3))

    #inequality constraints
    Aa = np.array([
        [0,mu,1],    # tau_xy + mu*sigma_y <= c_tau
        [0,mu,-1],   # -tau_xy + mu*sigma_y <= c_tau
        [0,1,0],     # sigma_y <= c_n

        [rgeom,mu,(1+mu*rgeom)],   # phi_d^h positive
        [rgeom,mu,-(1+mu*rgeom)],  # phi_d^h negative
        [mu,sgeom,(1+mu*sgeom)],   # phi_d^v positive
        [mu,sgeom,-(1+mu*sgeom)],  # phi_d^v negative

        [1,kgeom*mu,0],     # vertical +
        [1,-kgeom*mu,0]     # vertical -
    ], dtype=float)
    
    by = np.array([
        ctau,
        ctau,
        cn,
        ctau + rgeom*cn,
        ctau + rgeom*cn,
        ctau + sgeom*cn,
        ctau + sgeom*cn,
        cn + kgeom*ctau,
        cn + kgeom*ctau
    ], dtype=float)

    return Aseq, Aaeq, byeq, As, Aa, by


def setcon(nel, G):

    na = 8 # no alpha variables yet
    nr = 9 # yield constraints

    Ab  = lil_matrix((3*nel*nr, 9*nel + 1 + 3*nel*na))
    blc = np.zeros(3*nel*nr)
    buc = np.zeros(3*nel*nr)
    C   = [None]*(3*nel)

    for el in range(nel):
        for no in range(3):

            t    = G[el,0]
            ctau = G[el,1]
            cn   = G[el,2]
            mu   = G[el,3]
            hb   = G[el,4]
            lb   = G[el,5]
            f    = G[el,6]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_masonry(ctau, cn, mu, hb, lb, f)

            rp = (3*el + no)*nr
            r  = slice(rp, rp+nr)

            cp_beta = 9*el + no*3

            Ab[r, slice(cp_beta, cp_beta + 3)] = Aa

            blc[r] = -np.inf
            buc[r] = by

    return Ab.tocsr(), blc, buc, C