import paths
import proplot as pro
from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
from matplotlib.ticker import MaxNLocator
from matplotlib import patches
from photutils.profiles import RadialProfile
from scipy.optimize import minimize_scalar

pro.rc["cycle"] = "ggplot"
pro.rc["image.origin"] = "lower"
pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8

with fits.open(paths.data / "20230710_HD163296_resid_adc.fits") as hdul:
    cube = hdul[0].data
    header = hdul[0].header

titles = ("F610", "F670", "F720", "F760")

fig, axes = pro.subplots(nrows=2, ncols=2, width="3.5in", space=0.2, sharey=1, sharex=1)

plate_scale = header["PXSCALE"]
side_length = cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

## Plot and save
Jy_fact = (
    np.array((1.4e-6, 7e-7, 4.9e-7, 1.2e-6)) / header["PXAREA"] * 2
)  # Jy / sq.arcsec / (e-/s)
calib_data = cube * Jy_fact[:, None, None]


# PDI images
for ax, frame, title in zip(axes, calib_data, titles):
    im = ax.imshow(
        frame,
        extent=ext,
        cmap="magma",
        vmin=0,
        vmax=np.nanmax(frame),
        norm=simple_norm(frame, "sqrt"),
    )

    ax.text(0.025, 0.9, title, c="white", ha="left", fontsize=9, transform="axes")

bar_width_arc = 0.15
for ax in axes:
    rect = patches.Rectangle([-0.35, -0.35], bar_width_arc, 5e-3, color="white")
    ax.add_patch(rect)
    ax.text(
        -0.35 + bar_width_arc / 2,
        -0.33,
        f"{bar_width_arc:.02f}\"",
        c="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.scatter(
        [0], [0], m="+", ms=50, c="w", lw=0.75
    )

## sup title
axes.format(
    grid=False,
    xlim=(0.4, -0.4),
    ylim=(-0.4, 0.4),
    # xlabel=r'$\Delta$RA (")',
    # ylabel=r'$\Delta$DEC (")',
    facecolor="k",
)
# for ax in axes:
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

axes.format(yticks=[], xticks=[])

fig.savefig(
    paths.figures / "resid_adc.pdf",
    dpi=300,
)
