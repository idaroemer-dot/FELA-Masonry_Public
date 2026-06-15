import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from post.thesis_plot_style import THESIS_COLORS, set_thesis_plot_style


def principal_stresses(stress):
    sx, sy, txy = stress
    center = 0.5 * (sx + sy)
    radius = np.sqrt((0.5 * (sx - sy)) ** 2 + txy ** 2)
    angle = 0.5 * np.arctan2(txy, 0.5 * (sx - sy))
    return center + radius, center - radius, angle


def element_stress(x, el):
    stress = np.zeros(3)
    for gp in range(3):
        stress += np.asarray(x[9 * el + 3 * gp:9 * el + 3 * gp + 3], dtype=float)
    return stress / 3.0


def boundary_edges(T):
    count = {}
    for n1, n2, n3 in T[:, :3]:
        for edge in (tuple(sorted((n1, n2))), tuple(sorted((n2, n3))), tuple(sorted((n3, n1)))):
            count[edge] = count.get(edge, 0) + 1
    return [edge for edge, n in count.items() if n == 1]


def plotPS_contour(X, T, x, sfac, *, vmin=None, vmax=None, contour_type="max_abs", figsize=(5.2, 3.8), use_latex=False, savepath=None):
    set_thesis_plot_style(use_latex)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    element_values = np.zeros(T.shape[0])
    for el in range(T.shape[0]):
        s1, s2, _ = principal_stresses(element_stress(x, el))
        if contour_type == "s1":
            element_values[el] = s1 / 1e6
            label = r"$\sigma_1$ [MPa]"
        elif contour_type == "s2":
            element_values[el] = s2 / 1e6
            label = r"$\sigma_2$ [MPa]"
        else:
            element_values[el] = max(abs(s1), abs(s2)) / 1e6
            label = r"$\max(|\sigma_1|,|\sigma_2|)$ [MPa]"

    node_sum = np.zeros(X.shape[0])
    node_count = np.zeros(X.shape[0])
    for el, value in enumerate(element_values):
        for node in T[el, :3]:
            node_sum[node] += value
            node_count[node] += 1

    used = node_count > 0
    nodal_values = np.zeros(X.shape[0])
    nodal_values[used] = node_sum[used] / node_count[used]

    vmin = float(np.min(nodal_values[used])) if vmin is None else vmin
    vmax = float(np.max(nodal_values[used])) if vmax is None else vmax
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    cmap = LinearSegmentedColormap.from_list("thesis_rose", [np.ones(3), THESIS_COLORS["rose"], THESIS_COLORS["dark_rose"]])
    tri = mtri.Triangulation(X[:, 0], X[:, 1], triangles=T[:, :3])
    contour = ax.tripcolor(tri, nodal_values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.92)

    cbar = fig.colorbar(contour, ax=ax, fraction=0.030, pad=0.022, shrink=0.82, ticks=np.linspace(vmin, vmax, 5))
    cbar.set_label(label, rotation=90, labelpad=8)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(THESIS_COLORS["grey"])
    cbar.ax.tick_params(width=0.5, length=2.5, direction="in", labelsize=8, colors=THESIS_COLORS["dark"])

    for el in range(T.shape[0]):
        nodes = T[el, [0, 1, 2, 0]]
        ax.plot(X[nodes, 0], X[nodes, 1], color=THESIS_COLORS["lightgrey"], linewidth=0.22, alpha=0.55)

    for n1, n2 in boundary_edges(T):
        ax.plot([X[n1, 0], X[n2, 0]], [X[n1, 1], X[n2, 1]], color=THESIS_COLORS["dark"], linewidth=0.85, alpha=0.80)

    for el in range(T.shape[0]):
        xp = np.mean(X[T[el, :3], :], axis=0)
        s1, s2, angle = principal_stresses(element_stress(x, el))

        for stress, direction in [
            (s1, np.array([np.cos(angle), np.sin(angle)])),
            (s2, np.array([-np.sin(angle), np.cos(angle)])),
        ]:
            p1 = xp - sfac * direction * stress / 2.0
            p2 = xp + sfac * direction * stress / 2.0
            color = THESIS_COLORS["red"] if stress > 0 else THESIS_COLORS["blue"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=0.65, alpha=0.88)

    fig.tight_layout(pad=0.08)
    if savepath:
        fig.savefig(savepath)
    plt.show()
    return fig, ax
