import paths
import proplot as pro
from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
from matplotlib import patches
from skimage import transform

pro.rc["cycle"] = "ggplot"
pro.rc["image.origin"] = "lower"
pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8

with fits.open(paths.data / "20230707_HD204827_coll.fits") as hdul:
    cube = hdul[0].data
    header = hdul[0].header

titles = ("F610", "F670", "F720", "F760")

fig, axes = pro.subplots(
    nrows=2, ncols=2, width="3.5in", space=0.25, sharey=1, sharex=1
)

plate_scale = 5.9
side_length = cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)


## Plot and save
bs_fact = 0.3767
Jy_fact = (
    np.array((1.2e-6, 6.1e-7, 4.4e-7, 1e-6)) / bs_fact
)  # Jy / sq.arcsec / (e-/s)
calib_data = cube * Jy_fact[:, None, None]

# PDI images
for ax, frame, title in zip(axes, calib_data, titles):
    im = ax.imshow(
        frame,
        extent=ext,
        cmap="magma",
        vmin=0,
        vmax=np.nanmax(calib_data),
        norm=simple_norm(frame, "log"),
    )

    ax.text(0.025, 0.9, title, c="white", ha="left", fontsize=9, transform="axes")

# scale bar
bar_width_arc = 0.075
rect = patches.Rectangle([0.17, -0.17], -bar_width_arc, 5e-3, color="white")
axes[1, 0].add_patch(rect)
axes[1, 0].text(
    0.17 - bar_width_arc / 2,
    -0.15,
    f"{bar_width_arc*1e3:.0f} mas",
    c="white",
    ha="center",
    fontsize=8,
)

# compass rose
arrow_length = 0.04
delta = np.array((0, arrow_length))
axes[1, 1].plot((-0.12, delta[0] + -0.12), (-0.17, delta[1] + -0.17), color="w", lw=1)
axes[1, 1].text(
    delta[0] - 0.12,
    -0.17 + delta[1],
    "N",
    color="w",
    fontsize=7,
    ha="center",
    va="bottom",
)
delta = np.array((arrow_length, 0))
axes[1, 1].plot((-0.12, delta[0] + -0.12), (-0.17, delta[1] + -0.17), color="w", lw=1)
axes[1, 1].text(
    delta[0] - 0.115,
    -0.163 + delta[1],
    "E",
    color="w",
    fontsize=7,
    ha="right",
    va="top",
)



## sup title
axes.format(
    grid=False,
    xlim=(0.17 + 27e-3, -0.17 + 27e-3),
    ylim=(-0.17 - 25e-3, 0.17 - 25e-3),
    # xlabel=r'$\Delta$RA (")',
    # ylabel=r'$\Delta$DEC (")',
    facecolor="k",
)
# for ax in axes:
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

axes.format(yticks=[], xticks=[])

fig.savefig(
    paths.figures / "20230707_HD204827_binary_mosaic.pdf",
    dpi=300,
)
