from scipy.sparse import lil_matrix
import numpy as np

def Ab_masonry_ps(ftx, fcm, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max):
    """
    Local variables per constitutive point:

    beta = [sigma_x, sigma_y, tau_xy]

    alpha = [
        a1,    # 0: R
        a2,    # 1: tau_xy
        a3,    # 2: 0.5*(sigma_x - sigma_y)
        a4,    # 3: 0.5*(sigma_x + sigma_y)

        a5,    # 4: ftx - sigma_x
        a6,    # 5: -sigma_y
        a7,    # 6: sqrt(2)*tau_xy   (for I)

        a8,    # 7: A_x - sigma_x
        a9,    # 8: B_x + sigma_x
        a10    # 9: sqrt(2)*tau_xy   (for X)
    ]
    """

    # Parameters for X
    ls = 1.0 - 2.0 * (ftx / fcs) * (np.sin(phi_s) / (1.0 - np.sin(phi_s)))
    ms = 1.0 - 2.0 * (ftx / fcs) * (1.0 / (1.0 - np.sin(phi_s)))

    Ax = 0.5 * fcs * xi * (ls - ms)
    Bx = 0.5 * fcs * xi * (ls + ms) + fce * (1.0 - xi)

    # Parameters for XI
    # tau_xy <= c0 - cx*sigma_x - cy*sigma_y
    den = np.cos(omega_max - phi_l)
    if abs(den) < 1e-12:
        raise ValueError("cos(omega_max - phi_l) is too close to zero.")

    c0 = (
        0.5 * fcl * (1.0 - np.sin(phi_l)) * np.cos(omega_max)
        + 0.5 * fce * (1.0 - np.cos(phi_l)) * np.sin(omega_max)
    ) / den

    cx = (np.cos(phi_l) * np.sin(omega_max)) / den
    cy = (np.sin(phi_l) * np.cos(omega_max)) / den

    # Equalities
    Aseq = np.array([
        [ 0.0,  0.0, -1.0],            # a2 - tau_xy = 0
        [-0.5,  0.5,  0.0],            # a3 - 0.5*(sx-sy) = 0
        [-0.5, -0.5,  0.0],            # a4 - 0.5*(sx+sy) = 0
        [ 1.0,  0.0,  0.0],            # a5 + sx = ftx
        [ 0.0,  1.0,  0.0],            # a6 + sy = 0
        [ 0.0,  0.0,  np.sqrt(2.0)],   # -a7 + sqrt(2)*tau_xy = 0
        [ 1.0,  0.0,  0.0],            # a8 + sx = Ax
        [-1.0,  0.0,  0.0],            # a9 - sx = Bx
        [ 0.0,  0.0,  np.sqrt(2.0)],   # -a10 + sqrt(2)*tau_xy = 0
    ], dtype=float)

    Aaeq = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0],  # a2
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0],  # a3
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0],  # a4
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0,  0.0],  # a5
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0,  0.0],  # a6
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,  0.0],  # -a7
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0],  # a8
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  0.0],  # a9
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, -1.0],  # -a10
    ], dtype=float)

    byeq = np.array([
        0.0,
        0.0,
        0.0,
        ftx,
        0.0,
        0.0,
        Ax,
        Bx,
        0.0,
    ], dtype=float)

    # Inequalities
    # II:   a1 - a4 <= fcm
    # VIII: |tau_xy| <= 0.5*nu*fcm
    # XI:   +/-tau_xy <= c0 - cx*sigma_x - cy*sigma_y
    As = np.array([
    #    [ 0.0,  0.0,  0.0],   # I
        [ 0.0,  0.0,  0.0],   # II
        [ 0.0,  0.0,  1.0],   # VIIIa
        [ 0.0,  0.0, -1.0],   # VIIIb
        [ cx,   cy,   1.0],   # XIa
        [ cx,   cy,  -1.0],   # XIb
    ], dtype=float)

    Aa = np.array([
    #    [1.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # I
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # II
        [0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # VIIIa
        [0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # VIIIb
        [0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # XIa
        [0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # XIb
    ], dtype=float)

    by = np.array([
     #   ftx,
        fcm,
        0.5 * nu * fcm,
        0.5 * nu * fcm,
        c0,
        c0,
    ], dtype=float)

    return Aseq, Aaeq, byeq, As, Aa, by


def setcon_masonry(nel, G):
    """
    G[el,:] = [
        t,
        fcm,
        ftx,
        nu,
        fcs,
        fce,
        xi,
        phi_s,
        fcl,
        phi_l,
        omega_max
    ]
    """
    na = 10
    neq = 9
    nin = 5
    nr = neq + nin

    Ab  = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr, dtype=float)
    buc = np.zeros(3 * nel * nr, dtype=float)

    # 3 cones per constitutive point
    C = [None] * (3 * 3 * nel)

    for el in range(nel):
        for no in range(3):
            t         = G[el, 0]
            fcm       = G[el, 1]
            ftx       = G[el, 2]
            nu        = G[el, 3]
            fcs       = G[el, 4]
            fce       = G[el, 5]
            xi        = G[el, 6]
            phi_s     = G[el, 7]
            fcl       = G[el, 8]
            phi_l     = G[el, 9]
            omega_max = G[el,10]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_masonry_ps(
                ftx, fcm, nu, fcs, fce, xi, phi_s, fcl, phi_l, omega_max
            )

            rp = (3 * el + no) * nr
            r  = slice(rp, rp + nr)

            cp_beta = 9 * el + no * 3
            Ab[r, slice(cp_beta, cp_beta + 3)] = np.vstack((Aseq / t, As))

            cp_alfa = 9 * nel + 1 + (3 * el + no) * na
            Ab[r, slice(cp_alfa, cp_alfa + na)] = np.vstack((Aaeq, Aa))

            blc[r] = np.concatenate((byeq, -np.inf * np.ones_like(by)))
            buc[r] = np.concatenate((byeq, by))

            # Cone 1: a1 >= sqrt(a2^2 + a3^2)
            C[3 * (3 * el + no) + 0] = {
                "type": "MSK_CT_QUAD",
                "sub": cp_alfa + np.array([0, 1, 2])
            }

            # Cone 2: I -> 2*a5*a6 >= a7^2
            C[3 * (3 * el + no) + 1] = {
                "type": "MSK_CT_RQUAD",
                "sub": cp_alfa + np.array([4, 5, 6])
            }

            #Cone 3: X -> 2*a8*a9 >= a10^2
            C[3 * (3 * el + no) + 2] = {
                "type": "MSK_CT_RQUAD",
                "sub": cp_alfa + np.array([7, 8, 9])
            }

    return Ab.tocsr(), blc, buc, C
