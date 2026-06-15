import matplotlib.pyplot as plt
from post.thesis_plot_style import THESIS_COLORS, set_thesis_plot_style


def plot_topology(X, T, *, figsize=(4.8, 3.6), use_latex=False, savepath=None):
    set_thesis_plot_style(use_latex)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    edge_nodes = [0, 5, 1, 3, 2, 4, 0]
    for el in range(T.shape[0]):
        nodes = T[el, edge_nodes]
        ax.plot(X[nodes, 0], X[nodes, 1], color=THESIS_COLORS["dark"], linewidth=0.55)

    fig.tight_layout(pad=0.05)
    if savepath:
        fig.savefig(savepath)
    plt.show()
    return fig, ax
