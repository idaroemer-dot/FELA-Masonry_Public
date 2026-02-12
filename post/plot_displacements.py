import numpy as np
import matplotlib.pyplot as plt

cy = np.array([
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1]
])


def plotDof(X, T, y, supeq, nel, nno, scale):
    plt.figure(3)
    plt.axis("equal")
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")

#    print("2*nno =", 2*nno)
#    print("len(supeq) =", len(supeq))
#    print("expected free dofs =", 2*nno - len(supeq))
#    print("len(y) =", len(y))

    y = -np.asarray(y, dtype=float).ravel() * scale

    # expand reduced y to full y2 (size 2*nno), skipping constrained dofs in supeq
    supeq = set(np.asarray(supeq, dtype=int).tolist())
    y2 = np.zeros(2*nno, dtype=float)
    i = 0
    for dof in range(2*nno):
        if dof not in supeq:
            y2[dof] = y[i]
            i += 1

    nrp = 11
    xg = np.zeros((2, 3*(nrp-1) + 1), dtype=float)
    xr = np.zeros((2, 3*(nrp-1) + 1), dtype=float)

    for el in range(nel):
        k = 0
        for si in range(3):
            no1 = cy[1, si]
            no2 = cy[2, si]
            no3 = 3 + cy[0, si]  # mid-side node index in T

            x1 = X[T[el, no1], 0:2]
            x2 = X[T[el, no2], 0:2]

            v1 = y2[2*T[el, no1]:2*T[el, no1]+2]
            v2 = y2[2*T[el, no2]:2*T[el, no2]+2]
            v3 = y2[2*T[el, no3]:2*T[el, no3]+2]

            for j in range(nrp-1):
                k += 1
                s = j / (nrp - 1)

                xg[:, k-1] = (1-s)*x1 + s*x2
                xr[:, k-1] = (1-s)*x1 + s*x2 + (
                    2*(s-1)*(s-0.5)*v1
                    + 2*s*(s-0.5)*v2
                    - 4*s*(s-1)*v3
                ) * scale

        xg[:, -1] = xg[:, 0]
        xr[:, -1] = xr[:, 0]

        plt.plot(xg[0, :], xg[1, :], "k:")
        plt.plot(xr[0, :], xr[1, :], "k-")

    plt.show()
