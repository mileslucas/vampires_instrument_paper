import paths
import proplot as pro
from matplotlib import patches

import numpy as np

pro.rc["lines.linewidth"] = 1
pro.rc["font.size"] = 9

fig, axes = pro.subplots(width="3.5in", aspect=1)

fov = 3


def draw_fov(ax, x, y, title="", angles="", color="k", ls="-", offset=0, **kwargs):
    square = patches.Rectangle((x, y), fov, fov, ec=color, ls=ls, fill=False, lw=1)
    ax.add_patch(square)

    ax.text(
        x + 0.1, y + fov - 0.1 - offset, title, fontsize=7, c=color, va="top", ha="left"
    )
    ax.text(x + 0.15, y + 0.05, angles, fontsize=7, c=color, va="bottom", ha="left")

    ax.scatter(x + fov / 2, y + fov / 2, c=color, **kwargs)
    return axes


cycle = pro.Colormap("fire_r")(np.linspace(0.2, 0.6, 4))

draw_fov(
    axes[0],
    3,
    -3,
    title="Field 1",
    angles=r"(3$\varphi$, -$\varphi$)",
    color=cycle[0],
    m=".",
    zorder=800,
)
draw_fov(
    axes[0],
    -3,
    -3,
    title="Field 2",
    angles=r"($\varphi$, -$\varphi$)",
    color=cycle[1],
    m=".",
    zorder=700,
)
draw_fov(
    axes[0],
    -6,
    -3,
    title="Field 3",
    angles=r"(0, -$\varphi$)",
    color=cycle[2],
    m=".",
    zorder=600,
)
draw_fov(
    axes[0],
    -6,
    0,
    title="Field 4",
    angles=r"(0, $\varphi$)",
    color=cycle[3],
    m=".",
    zorder=500,
)
draw_fov(
    axes[0],
    -9,
    -3,
    title="Ghost 1",
    angles=r"(-$\varphi$, -$\varphi$)",
    color=cycle[1],
    ls="--",
    m="+",
    ms=30,
)
draw_fov(
    axes[0],
    -9,
    -3,
    title="Ghost 2",
    angles=r"(-$\varphi$, -$\varphi$)",
    offset=0.5,
    color=cycle[2],
    ls="--",
    m="x",
    ms=30,
)
draw_fov(
    axes[0],
    -6,
    3,
    title="Ghost 3",
    angles=r"(0, $2\varphi$)",
    color=cycle[2],
    ls="--",
    m="+",
    ms=30,
)

axes[0].scatter(0, 0, c="k", ms=60, m="+", zorder=900)
axes[0].text(0, 0.5, "Optical axis", c="0.3", fontsize=8, ha="center", va="bottom")
circ = patches.Circle((0, 0), np.hypot(6, 3), ec="0.3", lw=1, alpha=0.5, fill=False)
axes[0].add_patch(circ)

axes.format(
    xlim=(-9.2, 6.2),
    ylim=(-3.2, 6.2),
    xticks=[-9, -6, -3, 0, 3, 6],
    yticks=[-3, 0, 3, 6],
    xlabel=r'$\Delta x$ (")',
    ylabel=r'$\Delta y$ (")',
)

fig.savefig(paths.figures / "mbi_field_diagram.pdf", dpi=300)
