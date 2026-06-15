import numpy as np
import matplotlib.pyplot as plt
from post.thesis_plot_style import THESIS_COLORS, set_thesis_plot_style


CY = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])


def full_displacement_vector(y, support_dofs, number_nodes, scale):
    y = -np.asarray(y, dtype=float).ravel() * scale
    support_dofs = set(np.asarray(support_dofs, dtype=int).tolist())
    full_y = np.zeros(2 * number_nodes)

    j = 0
    for dof in range(2 * number_nodes):
        if dof not in support_dofs:
            full_y[dof] = y[j]
            j += 1
    return full_y


def plotDof(X, T, y, support_dofs, scale, *, show_undeformed=True, figsize=(4.8, 3.6), use_latex=False, savepath=None):
    set_thesis_plot_style(use_latex)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    full_y = full_displacement_vector(y, support_dofs, X.shape[0], scale)
    points_per_edge = 11

    for el in range(T.shape[0]):
        original = []
        deformed = []

        for side in range(3):
            n1 = CY[1, side]
            n2 = CY[2, side]
            n3 = 3 + CY[0, side]

            x1 = X[T[el, n1], :2]
            x2 = X[T[el, n2], :2]
            v1 = full_y[2 * T[el, n1]:2 * T[el, n1] + 2]
            v2 = full_y[2 * T[el, n2]:2 * T[el, n2] + 2]
            v3 = full_y[2 * T[el, n3]:2 * T[el, n3] + 2]

            for j in range(points_per_edge - 1):
                s = j / (points_per_edge - 1)
                original.append((1 - s) * x1 + s * x2)
                deformed.append((1 - s) * x1 + s * x2 + 2 * (s - 1) * (s - 0.5) * v1 + 2 * s * (s - 0.5) * v2 - 4 * s * (s - 1) * v3)

        original.append(original[0])
        deformed.append(deformed[0])
        original = np.asarray(original)
        deformed = np.asarray(deformed)

        if show_undeformed:
            ax.plot(original[:, 0], original[:, 1], ":", color=THESIS_COLORS["grey"], linewidth=0.65)
        ax.plot(deformed[:, 0], deformed[:, 1], color=THESIS_COLORS["dark"], linewidth=0.65)

    fig.tight_layout(pad=0.05)
    if savepath:
        fig.savefig(savepath)
    plt.show()
    return fig, ax
