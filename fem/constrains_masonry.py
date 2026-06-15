import numpy as np
from scipy.sparse import lil_matrix


def Ab_masonry_ps(
    ftx, fcm, nu, fcs, fts, fce, xi,
    phi_s, fcl, phi_l, omega_max,
    ix_mode="A17",
):
    ix_mode = ix_mode.upper()
    if ix_mode not in ("A17", "A18"):
        raise ValueError("ix_mode must be 'A17' or 'A18'.")

    s2 = np.sqrt(2.0)

    # head-joint parameters
    ls = 1.0 - 2.0 * (fts / fcs) * np.sin(phi_s) / (1.0 - np.sin(phi_s))
    ms = 1.0 - 2.0 * (fts / fcs) / (1.0 - np.sin(phi_s))

    Ax = 0.5 * fcs * xi * (ls - ms)
    Bx = 0.5 * fcs * xi * (ls + ms) + fce * (1.0 - xi)

    # staircase failure, written as: |tau_xy| <= c0 - cx*sigma_x - cy*sigma_y
    den = np.cos(omega_max - phi_l)
    if abs(den) < 1e-12:
        raise ValueError("cos(omega_max - phi_l) is too close to zero.")

    c0 = (
        0.5 * fcl * (1.0 - np.sin(phi_l)) * np.cos(omega_max)
        + 0.5 * fce * (1.0 - np.cos(phi_l)) * np.sin(omega_max)
    ) / den

    cx = np.cos(phi_l) * np.sin(omega_max) / den
    cy = np.sin(phi_l) * np.cos(omega_max) / den

    # linear bed-joint version
    cos_phi_l = np.cos(phi_l)
    if ix_mode == "A18" and abs(cos_phi_l) < 1e-12:
        raise ValueError("cos(phi_l) is too close to zero.")

    tan_phi_l = np.tan(phi_l)
    c_ix = 0.5 * fcl * (1.0 - np.sin(phi_l)) / cos_phi_l

    Aseq = np.array([
        [ 0.0,  0.0, -1.0],
        [-0.5,  0.5,  0.0],
        [-0.5, -0.5,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  s2 ],
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  s2 ],
        [ 0.0, -0.5,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=float)

    Aaeq = np.zeros((12, 13), dtype=float)
    Aaeq[0, 1] = 1.0
    Aaeq[1, 2] = 1.0
    Aaeq[2, 3] = 1.0
    Aaeq[3, 4] = 1.0
    Aaeq[4, 5] = 1.0
    Aaeq[5, 6] = -1.0
    Aaeq[6, 7] = 1.0
    Aaeq[7, 8] = 1.0
    Aaeq[8, 9] = -1.0
    Aaeq[9, 10] = 1.0
    Aaeq[10, 11] = -1.0
    Aaeq[11, 12] = 1.0

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
        0.5 * fcl,
        0.0,
        0.0,
    ], dtype=float)

    As_rows = [
        [0.0, 0.0,  0.0],       # II
        [0.0, 0.0,  1.0],       # VIII
        [0.0, 0.0, -1.0],       # VIII
        [cx,  cy,   1.0],       # XI
        [cx,  cy,  -1.0],       # XI
    ]

    Aa_rows = [
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0] * 13,
        [0.0] * 13,
        [0.0] * 13,
        [0.0] * 13,
    ]

    by = [
        fcm,
        0.5 * nu * fcm,
        0.5 * nu * fcm,
        c0,
        c0,
    ]

    if ix_mode == "A18":
        As_rows += [
            [0.0, tan_phi_l,  1.0],
            [0.0, tan_phi_l, -1.0],
        ]
        Aa_rows += [
            [0.0] * 13,
            [0.0] * 13,
        ]
        by += [c_ix, c_ix]

    return (
        Aseq,
        Aaeq,
        byeq,
        np.array(As_rows, dtype=float),
        np.array(Aa_rows, dtype=float),
        np.array(by, dtype=float),
    )


def setcon_masonry(nel, G, ix_mode="A17"):
    ix_mode = ix_mode.upper()
    if ix_mode not in ("A17", "A18"):
        raise ValueError("ix_mode must be 'A17' or 'A18'.")

    na = 13
    neq = 12
    nin = 5 if ix_mode == "A17" else 7
    nr = neq + nin
    ncones = 4 if ix_mode == "A17" else 3

    Ab = lil_matrix((3 * nel * nr, 9 * nel + 1 + 3 * nel * na))
    blc = np.zeros(3 * nel * nr, dtype=float)
    buc = np.zeros(3 * nel * nr, dtype=float)
    C = [None] * (ncones * 3 * nel)

    for el in range(nel):
        for no in range(3):
            (
                t,
                fcm,
                ftx,
                nu,
                fcs,
                fts,
                fce,
                xi,
                phi_s,
                fcl,
                phi_l,
                omega_max,
            ) = G[el, :12]

            Aseq, Aaeq, byeq, As, Aa, by = Ab_masonry_ps(
                ftx, fcm, nu, fcs, fts, fce, xi,
                phi_s, fcl, phi_l, omega_max,
                ix_mode=ix_mode,
            )

            rp = (3 * el + no) * nr
            r = slice(rp, rp + nr)

            cp_beta = 9 * el + 3 * no
            cp_alpha = 9 * nel + 1 + (3 * el + no) * na

            Ab[r, cp_beta:cp_beta + 3] = np.vstack((Aseq / t, As))
            Ab[r, cp_alpha:cp_alpha + na] = np.vstack((Aaeq, Aa))

            blc[r] = np.concatenate((byeq, -np.inf * np.ones(len(by))))
            buc[r] = np.concatenate((byeq, by))

            c0 = ncones * (3 * el + no)

            C[c0] = {
                "type": "MSK_CT_QUAD",
                "sub": cp_alpha + np.array([0, 1, 2]),
            }

            C[c0 + 1] = {
                "type": "MSK_CT_RQUAD",
                "sub": cp_alpha + np.array([4, 5, 6]),
            }

            if ix_mode == "A17":
                C[c0 + 2] = {
                    "type": "MSK_CT_RQUAD",
                    "sub": cp_alpha + np.array([10, 11, 12]),
                }

            C[c0 + ncones - 1] = {
                "type": "MSK_CT_RQUAD",
                "sub": cp_alpha + np.array([7, 8, 9]),
            }

    return Ab.tocsr(), blc, buc, C
