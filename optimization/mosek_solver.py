import mosek
import numpy as np
import scipy.sparse as sp


def solveopt(
    number_elements,
    global_equilibrium_reduced,
    global_load_const_reduced,
    global_load_var_reduced,
    Ab,
    blc,
    buc,
    C,
    force_scale=1e6,
):
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive.")

    n_var = Ab.shape[1]
    n_alpha = n_var - (9 * number_elements + 1)
    n_eq = global_equilibrium_reduced.shape[0]

    R_const = np.asarray(global_load_const_reduced, dtype=float).reshape(-1) / force_scale
    R_var = np.asarray(global_load_var_reduced, dtype=float).reshape(-1) / force_scale

    A_eq = sp.hstack(
        [
            sp.csr_matrix(global_equilibrium_reduced),
            sp.csr_matrix((-R_var).reshape(-1, 1)),
            sp.csr_matrix((n_eq, n_alpha)),
        ],
        format="csr",
    )

    A = sp.vstack([A_eq, sp.csr_matrix(Ab)], format="csr")
    n_con, n_var = A.shape

    blc_scaled = np.asarray(blc, dtype=float).ravel() / force_scale
    buc_scaled = np.asarray(buc, dtype=float).ravel() / force_scale

    blc_full = np.concatenate([R_const, blc_scaled])
    buc_full = np.concatenate([R_const, buc_scaled])

    lambda_index = 9 * number_elements
    inf = 1e30

    with mosek.Env() as env, env.Task() as task:
        task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end=""))

        task.appendvars(n_var)
        task.appendcons(n_con)

        for j in range(n_var):
            task.putvarbound(j, mosek.boundkey.fr, -inf, inf)
        task.putvarbound(lambda_index, mosek.boundkey.lo, 0.0, inf)

        task.putcj(lambda_index, 1.0)
        task.putobjsense(mosek.objsense.maximize)

        for i in range(n_con):
            lo = blc_full[i]
            up = buc_full[i]
            lo_ok = np.isfinite(lo)
            up_ok = np.isfinite(up)

            if lo_ok and up_ok:
                if abs(lo - up) < 1e-12:
                    task.putconbound(i, mosek.boundkey.fx, float(lo), float(up))
                else:
                    task.putconbound(i, mosek.boundkey.ra, float(lo), float(up))
            elif lo_ok:
                task.putconbound(i, mosek.boundkey.lo, float(lo), inf)
            elif up_ok:
                task.putconbound(i, mosek.boundkey.up, -inf, float(up))
            else:
                task.putconbound(i, mosek.boundkey.fr, -inf, inf)

        A_coo = A.tocoo()
        task.putaijlist(
            A_coo.row.astype(int),
            A_coo.col.astype(int),
            A_coo.data.astype(float),
        )

        for cone_group in C:
            if cone_group is None:
                continue
            if isinstance(cone_group, dict):
                cone_group = [cone_group]

            for cone in cone_group:
                if cone is None:
                    continue

                idx = np.asarray(cone["sub"], dtype=int).tolist()
                if cone["type"] == "MSK_CT_QUAD":
                    task.appendcone(mosek.conetype.quad, 0.0, idx)
                elif cone["type"] == "MSK_CT_RQUAD":
                    task.appendcone(mosek.conetype.rquad, 0.0, idx)
                else:
                    raise ValueError(f"Unknown cone type: {cone['type']}")

        task.optimize()
        solsta = task.getsolsta(mosek.soltype.itr)

        accepted = [
            mosek.solsta.optimal,
            mosek.solsta.prim_and_dual_feas,
            mosek.solsta.prim_feas,
        ]
        dual_valid = solsta in [
            mosek.solsta.optimal,
            mosek.solsta.prim_and_dual_feas,
        ]

        if solsta not in accepted and solsta != mosek.solsta.unknown:
            raise RuntimeError(f"MOSEK did not return a usable solution. Status: {solsta}")

        xx = np.zeros(n_var)
        yy = np.zeros(n_con)

        try:
            task.getxx(mosek.soltype.itr, xx)
            task.gety(mosek.soltype.itr, yy)
        except mosek.Error as e:
            raise RuntimeError(f"MOSEK returned no usable interior solution. Error: {e}")

    lambda_val = xx[lambda_index]
    x = xx[: 9 * number_elements + 1]
    alpha = xx[9 * number_elements + 1 :]

    x[: 9 * number_elements] *= force_scale
    alpha *= force_scale

    if dual_valid:
        y = yy[:n_eq] / force_scale
    else:
        y = None
        print(f"Warning: MOSEK status is {solsta}; dual vector is not reliable.")

    return x, alpha, y, lambda_val
