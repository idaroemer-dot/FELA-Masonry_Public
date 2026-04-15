import mosek
import scipy.sparse as sp
import numpy as np

def solveopt(number_elements, global_equilibrium_reduced, global_load_vector_reduced, Ab, blc, buc, C):
    # Total number of variables from Ab
    number_variables = Ab.shape[1]

    # Number of alpha variables
    n_alpha = number_variables - (9 * number_elements + 1)

    # Top equilibrium block
    m = global_equilibrium_reduced.shape[0]
    A_top = sp.hstack([
        sp.csr_matrix(global_equilibrium_reduced),
        sp.csr_matrix((-np.asarray(global_load_vector_reduced).reshape(-1, 1))),
        sp.csr_matrix((m, n_alpha))
    ], format="csr")

    # Full constraint matrix
    A = sp.vstack([A_top, sp.csr_matrix(Ab)], format="csr")
    number_constraints, number_variables = A.shape

    # Full bounds
    blc_full = np.concatenate([np.zeros(m), np.asarray(blc).ravel()])
    buc_full = np.concatenate([np.zeros(m), np.asarray(buc).ravel()])

    # Objective: maximize lambda
    c = np.zeros(number_variables)
    c[9 * number_elements] = 1.0

    with mosek.Env() as env, env.Task() as task:
        task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end=""))

        task.appendvars(number_variables)
        task.appendcons(number_constraints)

        inf = 1e30

        # Variable bounds: free
        for j in range(number_variables):
            task.putvarbound(j, mosek.boundkey.fr, -inf, inf)

        # Objective
        nonzero_c = np.nonzero(c)[0]
        for j in nonzero_c:
            task.putcj(int(j), float(c[j]))
        task.putobjsense(mosek.objsense.maximize)

        # Constraint bounds
        for i in range(number_constraints):
            lo = blc_full[i]
            up = buc_full[i]

            lo_fin = np.isfinite(lo)
            up_fin = np.isfinite(up)

            if lo_fin and up_fin:
                if abs(lo - up) < 1e-12:
                    task.putconbound(i, mosek.boundkey.fx, float(lo), float(up))
                else:
                    task.putconbound(i, mosek.boundkey.ra, float(lo), float(up))
            elif lo_fin and not up_fin:
                task.putconbound(i, mosek.boundkey.lo, float(lo), inf)
            elif not lo_fin and up_fin:
                task.putconbound(i, mosek.boundkey.up, -inf, float(up))
            else:
                task.putconbound(i, mosek.boundkey.fr, -inf, inf)

        # Sparse A matrix
        Acoo = A.tocoo()
        task.putaijlist(
            Acoo.row.astype(int),
            Acoo.col.astype(int),
            Acoo.data.astype(float)
        )

        # Cones
        for cone_group in C:
            if cone_group is None:
                continue

            # Handle both old format (single dict) and new format (list of dicts)
            if isinstance(cone_group, dict):
                cone_group = [cone_group]

            for cone in cone_group:
                if cone is None:
                    continue

                idx = np.asarray(cone["sub"], dtype=int).tolist()
                cone_type = cone["type"]

                if cone_type == "MSK_CT_QUAD":
                    task.appendcone(mosek.conetype.quad, 0.0, idx)

                elif cone_type == "MSK_CT_RQUAD":
                    task.appendcone(mosek.conetype.rquad, 0.0, idx)

                else:
                    raise ValueError(f"Unknown cone type: {cone_type}")

        # Solve
        task.optimize()
        solsta = task.getsolsta(mosek.soltype.itr)

        if solsta not in [
            mosek.solsta.optimal,
            mosek.solsta.prim_and_dual_feas,
            mosek.solsta.prim_feas
        ]:
            raise RuntimeError(f"MOSEK did not return a usable solution. Status: {solsta}")

        xx = np.zeros(number_variables)
        yy = np.zeros(number_constraints)

        task.getxx(mosek.soltype.itr, xx)
        task.gety(mosek.soltype.itr, yy)

        lambda_val = xx[9 * number_elements]

        x = xx[:(9 * number_elements + 1)]     # stress + lambda
        alpha = xx[(9 * number_elements + 1):] # alle alpha-variabler
        y = yy[:m]

        return x, alpha, y, lambda_val
