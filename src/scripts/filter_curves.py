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
from matplotlib import ticker

pro.rc["font.size"] = 8
pro.rc["title.size"] = 8

fig, axes = pro.subplots(
    nrows=5,
    width="3.5in",
    height="4.7in",
    space=0,
    sharey=1,
    hratios=(0.2125, 0.225, 0.2125, 0.2125, 0.15),
)

waveset = np.arange(550, 800, 0.1) * u.nm


def plot_filter(ax, wave, filt, name, color, space=0.1, plot_ave=True, **kwargs):
    transmission = filt(wave)
    mask = transmission >= 0.5 * transmission.max()
    waveset = wave[mask]
    lam_ave = filt.avgwave(waveset).to(u.nm)
    max_trans = filt.tpeak(waveset)
    line_height = max_trans + space
    ax.plot(wave.value, transmission, c=color)
    if plot_ave:
        ax.vlines(lam_ave.value, 0, line_height, lw=0.75, ls="--", c=color)
    else:
        ax.vlines(
            lam_ave.value, max_trans + 2e-2, line_height, lw=0.75, ls="--", c=color
        )
    edge_bbox_pars = dict(facecolor=pro.rc["axes.facecolor"], linewidth=0, pad=0)
    kwds = {
        "ha": "center",
        "size": 7,
        "c": color,
        "bbox": edge_bbox_pars,
    }
    kwds.update(kwargs)
    ax.text(lam_ave.value, line_height + 1e-2, name, **kwds)


def plot_nd_filter(ax, wave, filt, name, color, space=1.4):
    transmission = filt(wave)
    mask = transmission >= 0.5 * transmission.max()
    waveset = wave[mask]
    lam_ave = filt.avgwave(waveset).to(u.nm)
    max_trans = filt.tpeak(waveset)
    line_height = max_trans * space
    ax.plot(wave.value, transmission, c=color)
    edge_bbox_pars = dict(facecolor=pro.rc["axes.facecolor"], linewidth=0, pad=0)
    ax.text(
        lam_ave.value,
        line_height,
        name,
        ha="center",
        size=7,
        c=color,
        bbox=edge_bbox_pars,
    )


open_sp_elem = load_vampires_filter("Open")
plot_filter(axes[0], waveset, open_sp_elem, "Open", color="0.3", space=0.05)
axes[0].format(title="Input beam", titleloc="ul")
axes[0].text(
    0.18,
    0.3,
    "AO188\ndichroic",
    transform="axes",
    ha="center",
    va="bottom",
    fontsize=7,
    c="0.4",
)
axes[0].text(
    0.88,
    0.2,
    "PyWFS\ndichroic",
    transform="axes",
    ha="center",
    va="bottom",
    fontsize=7,
    c="0.4",
)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, len(VAMPIRES_STD_FILTERS) - 1))
cmap = dict(zip(("625-50", "675-50", "725-50", "750-50", "775-50"), cycle))
for i, filt in enumerate(sorted(VAMPIRES_STD_FILTERS - {"Open"})):
    sp_elem = load_vampires_filter(filt)
    if filt == "750-50":
        space = 0.17
    else:
        space = 0.05
    if filt == "Open":
        color = "0.3"
    else:
        color = cmap[filt]
    plot_filter(axes[1], waveset, sp_elem, filt, color=color, space=space)
axes[1].format(title="Standard filters", titleloc="ul")

# plot MBI filters
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, len(VAMPIRES_MBI_FILTERS)))
cmap = dict(zip(("F610", "F670", "F720", "F760"), cycle))
for i, filt in enumerate(sorted(VAMPIRES_MBI_FILTERS)):
    sp_elem = load_vampires_filter(filt)
    space = 0.05
    plot_filter(axes[2], waveset, sp_elem, filt, color=cmap[filt], space=space)
axes[2].format(title="MBI filters", titleloc="ul")

# plot NB filters
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, len(VAMPIRES_NB_FILTERS)))
cmap = dict(zip(("Ha-Cont", "Halpha", "SII", "SII-Cont"), cycle))
for i, filt in enumerate(("Ha-Cont", "Halpha", "SII", "SII-Cont")):
    sp_elem = load_vampires_filter(filt)
    space = 0.08
    ha = "center"
    if filt == "Ha-Cont":
        ha = "right"
    elif filt == "SII-Cont":
        ha = "left"

    color = cmap[filt]
    name = filt.replace("Halpha", r"H$\alpha$").replace("Ha", r"H$\alpha$")
    plot_filter(axes[3], waveset, sp_elem, name, color=color, space=space, ha=ha)
axes[3].format(title="NB filters", titleloc="ul")


cmap = dict(zip(("ND10", "ND25"), ("0.6", "0.3")))
for i, filt in enumerate(("ND10", "ND25")):
    sp_elem = load_vampires_filter(filt)
    color = cmap[filt]
    plot_nd_filter(
        axes[4],
        waveset,
        sp_elem,
        filt,
        color=color,
    )
axes[4].format(
    yscale="log", yformatter="log", ylim=(1e-3, 8e-1), title="ND filters", titleloc="ul"
)

axes.format(ylabel="transmission", xlabel=r"wavelength (nm)", titlesize=7)

axes[:-1].format(
    ylocator=ticker.FixedLocator((0, 0.25, 0.5, 0.75, 1)),  # (6, prune="both"),
    ylim=(-0.05, 1.35),
    xlim=(560, 785),
)
axes[0].format(ylim=(None, 1.25))
axes[2].format(ylim=(None, 1.2))
axes[3].format(ylim=(None, 1.25))

# for ax in axes[2:]:

# save output
fig.savefig(paths.figures / "filter_curves.pdf", dpi=300)
