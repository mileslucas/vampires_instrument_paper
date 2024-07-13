import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from astropy.visualization import simple_norm
from photutils.profiles import RadialProfile

pro.rc["legend.fontsize"] = 6
pro.rc["legend.frameon"] = True
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["grid"] = False
pro.rc["cmap"] = "magma"
pro.rc["image.origin"] = "lower"


frame, hdr = fits.getdata(paths.data / "20230707_HD191195_frame.fits", header=True)

plate_scale = 5.9
side_length = frame.shape[-1] * plate_scale * 1e-3 / 2
ext = (-side_length, side_length, -side_length, side_length)

fig, axes = pro.subplots(width="3.5in")

ny = frame.shape[-2]
nx = frame.shape[-1]
center = (ny - 1) / 2, (nx - 1) / 2
Ys, Xs = np.ogrid[: frame.shape[-2], : frame.shape[-1]]


reselem = np.rad2deg(720e-9 / 7.95) * 3.6e6
cutoff = 22.5 * hdr["RESELEM"]
norm_frame = frame[2] / np.nanmax(frame[2])

axes[0].imshow(
    norm_frame,
    norm=simple_norm(norm_frame, "log"),
    extent=ext,
    cmap="magma",
    vmin=-5e-5,
    vmax=0.12 * np.nanmax(norm_frame),
)

# axes[0].text(
#     0.03, 0.97, "F720", c="w", fontsize=9, va="top", ha="left", transform="axes"
# )
# axes[0].text(
#     0.03, 0.03, f'{hdr["NCOADD"]} x {hdr["EXPTIME"]*1e3:3.01f} ms', c="w", fontsize=7, va="bottom", ha="left", transform="axes"
# )

axes.format(
    xlim=(-1, 1),
    ylim=(-1, 1),
    xlabel=r'$\Delta$x (")',
    ylabel=r'$\Delta$y (")',
)

# bar_width_arc = 0.2
# rect = patches.Rectangle([0.7, -0.9], bar_width_arc, 3e-3, color="white")
# axes[0].add_patch(rect)
# axes[0].text(
#     0.7 + bar_width_arc / 2,
#     -0.88,
#     f'{bar_width_arc:.02f}"',
#     c="white",
#     ha="center",
#     va="bottom",
#     fontsize=7,
# )

## Filter label
axes[0].text(0.02, 0.02, r"$\lambda =$720 nm", c="w", ha="left", va="bottom", fontsize=9, transform="axes")

## speckle labels
text_kwargs = dict(fontsize=7, c="w", ha="center", va="bottom")
axes[0].text(0.66, 0.66, "G1", **text_kwargs)
axes[0].text(0.6, -0.6, "G2", **text_kwargs)
axes[0].text(-0.65, -0.52, "G3", **text_kwargs)
axes[0].text(-0.61, 0.73, "G4", **text_kwargs)


## spider labels
axes[0].text(0.02, 0.56, "S1", **text_kwargs)
axes[0].text(-0.4, 0.09, "S2", **text_kwargs)

## residual speckles
axes[0].text(-0.2, -0.23, "QS", **text_kwargs)

## dark hole
axes[0].text(0.5, 0.06, "CR", **text_kwargs)


# axes.format(xtickloc="none", ytickloc="none")
fig.savefig(paths.figures / "onsky_psfs.pdf", dpi=300)
