import mosek
import scipy.sparse as sp
import numpy as np

def solveopt(number_elements, global_equilibrium_reduced, global_load_vector_reduced, Ab, blc, buc, C):
    # Build A matrix for constraints
    m = global_equilibrium_reduced.shape[0]
    n_alpha = 3 * number_elements * 8
    A_top = sp.hstack([
        sp.csr_matrix(global_equilibrium_reduced),
        sp.csr_matrix((-np.asarray(global_load_vector_reduced).reshape(-1, 1))),
        sp.csr_matrix((m, n_alpha))
    ], format="csr")
    A = sp.vstack([A_top, sp.csr_matrix(Ab)], format="csr")

    number_constraints, number_variables = A.shape

    # Constraint bounds
    blc_full = np.concatenate([np.zeros(m), np.asarray(blc).ravel()])
    buc_full = np.concatenate([np.zeros(m), np.asarray(buc).ravel()])

    # Objective: maximize variable (9*nel) 
    c = np.zeros(number_variables)
    c[9 * number_elements] = 1.0


    with mosek.Env() as env, env.Task() as task:

        task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end=""))

        task.appendvars(number_variables)
        task.appendcons(number_constraints)

        # Variable bounds: free
        inf = 1e30
        for j in range(number_variables):
            task.putvarbound(j, mosek.boundkey.fr, -inf, inf)

        # Objective
        for j in range(number_variables):
            if c[j] != 0.0:
                task.putcj(j, c[j])
        task.putobjsense(mosek.objsense.maximize)

        # Constraint bounds
        for i in range(number_constraints):
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
        # for cone in C:
        #     idx = np.asarray(cone["sub"], dtype=int).tolist()  
        #     task.appendcone(mosek.conetype.quad, 0.0, idx)

        for cone in C:
            if cone is None:
                continue
            if cone["type"] == "MSK_CT_QUAD":
                idx = np.asarray(cone["sub"], dtype=int).tolist()
                task.appendcone(mosek.conetype.quad, 0.0, idx)

        # Solve
        task.optimize()

        xx = np.zeros(number_variables)
        yy = np.zeros(number_constraints)
        task.getxx(mosek.soltype.itr, xx)
        task.gety(mosek.soltype.itr, yy)
        lambda_val = task.getprimalobj(mosek.soltype.itr)


        x = xx[:(9 * number_elements + 1)]
        y = yy[:m]
        return x, y, lambda_val