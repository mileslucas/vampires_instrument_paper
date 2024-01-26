from scexao_etc.filters import (
    load_vampires_filter,
    VAMPIRES_STD_FILTERS,
    VAMPIRES_MBI_FILTERS,
    VAMPIRES_NB_FILTERS,
)
import paths
import proplot as pro
import numpy as np
import astropy.units as u
from matplotlib.ticker import MaxNLocator, FixedLocator

pro.rc["font.size"] = 8

fig, axes = pro.subplots(
    [[1, 1], [2, 2], [3, 4]],
    width="3.5in",
    height="5in",
    wspace=0,
    hspace=(0, 2),
    sharex=1,
)

waveset = np.arange(550, 800, 0.1) * u.nm


def plot_filter(ax, wave, filt, name, color, space=0.1, plot_ave=True):
    transmission = filt(wave)
    mask = transmission >= 0.5 * transmission.max()
    waveset = wave[mask]
    lam_ave = filt.avgwave(waveset).to(u.nm)
    max_trans = filt.tpeak(waveset)
    line_height = max_trans + space
    ax.plot(wave.value, transmission, c=color)
    if plot_ave:
        ax.vlines(lam_ave.value, 0, line_height, lw=1, ls="--", c=color)
    else:
        ax.vlines(lam_ave.value, max_trans + 2e-2, line_height, lw=1, ls="--", c=color)
    edge_bbox_pars = dict(facecolor=pro.rc["axes.facecolor"], linewidth=0, pad=0)
    ax.text(
        lam_ave.value,
        line_height + 1e-2,
        name,
        ha="center",
        size=7,
        c=color,
        bbox=edge_bbox_pars,
    )


# plot standard filters
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, len(VAMPIRES_STD_FILTERS) - 1))
cmap = dict(zip(("625-50", "675-50", "725-50", "750-50", "775-50"), cycle))
for i, filt in enumerate(VAMPIRES_STD_FILTERS):
    sp_elem = load_vampires_filter(filt)
    if filt in ("Open", "750-50"):
        space = 0.11
    else:
        space = 0.03
    if filt == "Open":
        color = "0.3"
    else:
        color = cmap[filt]
    plot_filter(axes[0], waveset, sp_elem, filt, color=color, space=space)
axes[0].format(ylim=(-0.02, 1.25), title="Standard Filters", titleloc="ul")

# plot MBI filters
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, len(VAMPIRES_MBI_FILTERS)))
cmap = dict(zip(("F610", "F670", "F720", "F760"), cycle))
for i, filt in enumerate(VAMPIRES_MBI_FILTERS):
    sp_elem = load_vampires_filter(filt)
    space = 0.03
    plot_filter(axes[1], waveset, sp_elem, filt, color=cmap[filt], space=space)
axes[1].format(ylim=(-0.02, 1.15), title="MBI Filters", titleloc="ul")

# plot NB filters
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, len(VAMPIRES_NB_FILTERS)))
cmap = dict(zip(("Ha-Cont", "Halpha", "SII", "SII-Cont"), cycle))
for i, filt in enumerate(VAMPIRES_NB_FILTERS):
    sp_elem = load_vampires_filter(filt)
    space = 0.03
    if "Ha" in filt:
        ax = axes[2]
    else:
        ax = axes[3]
    color = cmap[filt]
    name = filt.replace("Halpha", r"H$\alpha$").replace("Ha", r"H$\alpha$")
    plot_filter(
        ax,
        waveset,
        sp_elem,
        name,
        color=color,
        space=space,
        # plot_ave=False,
    )
axes[2:].format(ylim=(-0.02, 1.15), xlabel="wavelength (nm)")
axes[2].format(xlim=(644, 660), title=r"H$\alpha$", titleloc="uc")
axes[3].format(xlim=(668, 686), title=r"SII", titleloc="uc")
axes[:2].format(xlim=(575, 780))
axes.format(ylabel="transmission", titlesize=9)

for ax in axes:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

axes[2].xaxis.set_major_locator(FixedLocator([647.6, 652.0, 656.3]))
axes[3].xaxis.set_major_locator(FixedLocator([672.7, 677.1, 681.5]))
# for ax in axes[2:]:

# save output
fig.savefig(paths.figures / "filter_curves.pdf", dpi=300)
