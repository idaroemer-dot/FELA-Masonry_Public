import numpy as np
import matplotlib.pyplot as plt

def plotPS(X, T, x, nel, sfac):
    plt.figure(2)
    plt.title("Principal stresses")
    plt.axis("equal")
    ldof = [0, 1, 2, 0]  # local dof indices for plotting element edges

    for el in range(nel):
        plt.plot(X[T[el, ldof], 0], X[T[el, ldof], 1], "b-")

    for el in range(nel):
        xp = np.mean(X[T[el, :], :], axis=0)

        s = np.zeros(3)
        for i in range(3):
            num = 9*el + 3*i
            s += np.asarray(x[num:num+3], dtype=float)
        s /= 3.0

        C = 0.5*(s[0] + s[1])
        ds = 0.5*(s[0] - s[1])
        R  = np.sqrt(ds*ds + s[2]*s[2])
        s1 = C + R
        s2 = C - R

        phi = 0.5*np.arctan2(s[2], ds)

        n = np.array([np.cos(phi), np.sin(phi)]) * sfac
        p1 = xp - n * (s1/2.0)
        p2 = xp + n * (s1/2.0)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-" if s1 > 0 else "b-")

        n = np.array([-n[1], n[0]])
        p1 = xp - n * (s2/2.0)
        p2 = xp + n * (s2/2.0)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-" if s2 > 0 else "b-")

    plt.show()
