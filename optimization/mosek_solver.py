import mosek
import scipy.sparse as sp
import numpy as np

def solveopt(nel, Hsup, Rsup, Ab, blc, buc, C):
    # Build A = [Hsup  -Rsup  0;
    #            Ab           ]
    m = Hsup.shape[0]
    n_alpha = 3 * nel * 8
    A_top = sp.hstack([
        sp.csr_matrix(Hsup),
        sp.csr_matrix((-np.asarray(Rsup).reshape(-1, 1))),
        sp.csr_matrix((m, n_alpha))
    ], format="csr")
    A = sp.vstack([A_top, sp.csr_matrix(Ab)], format="csr")

    ncon, nvar = A.shape

    # Constraint bounds
    blc_full = np.concatenate([np.zeros(m), np.asarray(blc).ravel()])
    buc_full = np.concatenate([np.zeros(m), np.asarray(buc).ravel()])

    # Objective: maximize variable (9*nel) (0-based)
    c = np.zeros(nvar)
    c[9 * nel] = 1.0

    with mosek.Env() as env, env.Task() as task:
        task.appendvars(nvar)
        task.appendcons(ncon)

        # Variable bounds: free
        inf = 1e30
        for j in range(nvar):
            task.putvarbound(j, mosek.boundkey.fr, -inf, inf)

        # Objective
        for j in range(nvar):
            if c[j] != 0.0:
                task.putcj(j, c[j])
        task.putobjsense(mosek.objsense.maximize)

        # Constraint bounds
        for i in range(ncon):
            lo, up = blc_full[i], buc_full[i]
            if np.isfinite(lo) and np.isfinite(up):
                if abs(lo - up) < 1e-12:
                    task.putconbound(i, mosek.boundkey.fx, lo, up)
                else:
                    task.putconbound(i, mosek.boundkey.ra, lo, up)
            elif np.isfinite(lo) and not np.isfinite(up):
                task.putconbound(i, mosek.boundkey.lo, lo, inf)
            elif not np.isfinite(lo) and np.isfinite(up):
                task.putconbound(i, mosek.boundkey.up, -inf, up)
            else:
                task.putconbound(i, mosek.boundkey.fr, -inf, inf)

        # Put A (sparse)
        Acoo = A.tocoo()
        task.putaijlist(Acoo.row.astype(int), Acoo.col.astype(int), Acoo.data.astype(float))

        # Cones (C entries are dicts: {"type":"MSK_CT_QUAD","sub": array_of_indices})
        for cone in C:
            idx = np.asarray(cone["sub"], dtype=int).tolist()  # already 0-based
            task.appendcone(mosek.conetype.quad, 0.0, idx)

        # Solve
        task.optimize()

        xx = np.zeros(nvar)
        yy = np.zeros(ncon)
        task.getxx(mosek.soltype.itr, xx)
        task.gety(mosek.soltype.itr, yy)

        x = xx[:(9 * nel + 1)]
        y = yy[:m]
        return x, y