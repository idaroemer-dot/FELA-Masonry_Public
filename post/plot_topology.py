import matplotlib.pyplot as plt
import numpy as np

def plot_topology(X, T, nel, nno):
    # plot geometry
    plt.figure(1)
    plt.title('Element topologi')
    plt.axis('equal')
    ldof = [0, 5, 1, 3, 2, 4, 0]

    for el in range(nel):
        plt.plot(X[T[el, ldof], 0], X[T[el, ldof], 1], 'b-')

    for no in range(nno):
        plt.text(X[no, 0], X[no, 1], str(no + 1),
                 color='blue', backgroundcolor=(0.7, 0.7, 0.7))

    for el in range(nel):
        xp = np.mean(X[T[el, :], :], axis=0)
        plt.text(xp[0], xp[1], str(el + 1), color='black')

    plt.show()