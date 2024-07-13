import paths
import proplot as pro
from matplotlib import patches

import numpy as np

pro.rc["axes.grid"] = False
pro.rc["axes.spines.left"] = False
pro.rc["axes.spines.right"] = False
pro.rc["axes.spines.top"] = False
pro.rc["axes.spines.bottom"] = False
pro.rc["lines.linewidth"] = 1
pro.rc["cycle"] = "ggplot"

fig, axes = pro.subplots(width="3.5in", height="2.4in")

theta = np.deg2rad(10)
cost = np.cos(theta)
sint = np.sin(theta)
length = 3
offset = 3
beam_length = 3
ax_length = 2
aspect = (4 - -3.5) / (6 - -5.5)
arc_size = 2

# 1st dichro
xs = [-length * sint, length * sint]
ys = [length * cost, -length * cost]
axes[0].plot(xs, ys, color="0.3", lw=2)
axes[0].plot([0, 0], [ax_length, -ax_length], color="0.3")
axes[0].text(0, length + 0.15, "D1", color="0.3", fontsize=10, ha="center", va="bottom")
arc = patches.Arc(
    (0, 0),
    arc_size * aspect,
    arc_size,
    theta1=-90,
    theta2=-90 + np.rad2deg(theta),
    color="0.3",
    lw=0.75,
    fill=False,
)
axes[0].add_patch(arc)
axes[0].text(0.03, -1.4, r"$\theta$", c="0.3", fontsize=10, ha="left", va="top")

# mirror
xs = [offset, offset]
ys = [length, -length]
axes[0].plot(xs, ys, color="0.3", lw=4)
axes[0].text(
    offset, length + 0.15, "M", color="0.3", fontsize=10, ha="center", va="bottom"
)

# incoming beam
axes[0].arrow(
    -beam_length,
    0,
    beam_length,
    0,
    color="C0",
    head_width=0.3,
    overhang=0.5,
    lw=1,
    length_includes_head=True,
)
axes[0].text(-0.4, 0.2, "Input", c="C0", fontsize=9, ha="right", va="bottom")


# figure 1, reflect off first surface
cos2t = np.cos(2 * theta)
sin2t = np.sin(2 * theta)
arc = patches.Arc(
    (0, 0),
    arc_size * 1.5 * aspect,
    arc_size * 1.5,
    theta1=180,
    theta2=200,
    color="C5",
    lw=0.75,
    fill=False,
)
axes[0].add_patch(arc)
axes[0].text(-1.2, -0.1, r"$2\theta$", c="C5", fontsize=9, ha="right", va="top")
axes[0].arrow(
    0,
    0,
    -beam_length * 0.6 * cos2t,
    -beam_length * 0.6 * sin2t,
    head_width=0.3,
    overhang=0.5,
    lw=1,
    color="C5",
)
axes[0].text(
    -beam_length * 0.6 * cos2t - 0.2,
    -beam_length * 0.6 * sin2t - 0.45,
    "PSF 1",
    c="C5",
    fontsize=9,
    ha="right",
    va="top",
)


# figure 2, reflect off second surface
xs = [0, offset, 0]
ys = [0, 0, 0]
axes[0].plot(xs, ys, color="C1")
axes[0].arrow(
    xs[-1],
    ys[-1],
    -beam_length * 0.6,
    0,
    color="C1",
    lw=1,
    head_width=0.3,
    overhang=0.5,
    zorder=100,
)
axes[0].text(
    xs[-1] - beam_length * 0.7,
    ys[-1] - 0.3,
    "PSF 2",
    c="C1",
    fontsize=9,
    ha="right",
    va="top",
)


# figure 3, reflect off second surface with internal reflection
l = beam_length * 1.8
xs = [0, offset, offset - l * cos2t]
ys = [0, np.tan(2 * theta) * offset, np.tan(2 * theta) * offset + l * sin2t]
# add in final ray
axes[0].plot(xs, ys, color="C1", ls="--")
axes[0].arrow(
    xs[-1],
    ys[-1],
    -0.01 * cos2t,
    0.01 * sin2t,
    head_width=0.3,
    overhang=0.5,
    color="C1",
    lw=1,
    length_includes_head=True,
)
axes[0].text(
    xs[-1] + 0.3, ys[-1] - 0.55, "Ghost", c="C1", fontsize=9, ha="center", va="top"
)
arc = patches.Arc(
    (0, 0),
    arc_size * 1.5 * aspect,
    arc_size * 1.5,
    theta1=0,
    theta2=2 * np.rad2deg(theta),
    color="C1",
    lw=0.75,
    fill=False,
)
axes[0].add_patch(arc)
axes[0].text(1.2, 0, r"$2\theta$", c="C1", fontsize=9, ha="left", va="bottom")
arc = patches.Arc(
    (offset, ys[1]),
    arc_size * 1.5 * aspect,
    arc_size * 1.5,
    theta1=160,
    theta2=180,
    color="C1",
    lw=0.75,
    fill=False,
)
axes[0].add_patch(arc)
axes[0].line([offset, offset - 1.6], [ys[1], ys[1]], c="C1", lw=0.75)
axes[0].text(
    offset - 1.6, ys[1] + 0.02, r"$2\theta$", c="C1", fontsize=9, ha="left", va="bottom"
)

axes.format(
    xlim=(-3.2, 3),
    ylim=(-3.2, 4),
    xticks=[],
    yticks=[],
)

fig.savefig(paths.figures / "mbi_ghost_raytrace.pdf", dpi=300)
