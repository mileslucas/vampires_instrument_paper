import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from astropy.nddata import Cutout2D
from matplotlib import patches

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"

names = ("F610", "F670", "F720", "F760")
resid_cube, header = fits.getdata(paths.data / "20230711_HD1160" / "20230711_HD1160_GreeDS20_speccube.fits", header=True)

sdi_names = ("ADI+Mean", "ADI+SDI")

adi_frame, header = fits.getdata(paths.data / "20230711_HD1160" / "20230711_HD1160_GreeDS20_all.fits", header=True)
adisdi_frame, header = fits.getdata(paths.data / "20230711_HD1160" / "20230711_HD1160_GreeDS20_ASDI.fits", header=True)

plate_scale = 5.9  # mas / px
fig, axes = pro.subplots([[1, 2, 5], [3, 4, 6]], width="7in", wspace=[0.25, 0.75], hspace=0.25, spanx=False)

exp_sep = 135.18
exp_pa = 245.1
cy, cx = np.array(resid_cube.shape[-2:]) / 2 - 0.5
dx = -exp_sep * np.sin(np.radians(exp_pa))
dy = exp_sep * np.cos(np.radians(exp_pa))


for i, ax in enumerate(axes[:-2]):
    frame = resid_cube[i]
    cutout = Cutout2D(frame, (cx + dx, cy + dy), 40)
    side_length = np.array(cutout.shape) * plate_scale / 1e3
    yext, xext = cutout.bbox_original
    ext = (
        (cx - xext[0]) * plate_scale / 1e3,
        (cx - xext[1]) * plate_scale / 1e3,
        (yext[0] - cy) * plate_scale / 1e3,
        (yext[1] - cy) * plate_scale / 1e3
    )
    ax.imshow(cutout.data, extent=ext, cmap="magma", origin="lower")
    ax.text(0.05, 0.95, names[i], c="w", va="top", ha="left", fontsize=9, transform="axes")


cutout = Cutout2D(adi_frame, (cx + dx, cy + dy), 40)
yext, xext = cutout.bbox_original
ext = (
    (cx - xext[0]) * plate_scale / 1e3,
    (cx - xext[1]) * plate_scale / 1e3,
    (yext[0] - cy) * plate_scale / 1e3,
    (yext[1] - cy) * plate_scale / 1e3
)
axes[4].imshow(cutout.data, extent=ext, cmap="magma", origin="lower")
axes[4].text(0.05, 0.95, sdi_names[0], c="w", va="top", ha="left", fontsize=9, transform="axes")

cutout = Cutout2D(adisdi_frame, (cx + dx, cy + dy), 40)
axes[5].imshow(cutout.data, extent=ext, cmap="magma", origin="lower")
axes[5].text(0.05, 0.95, sdi_names[1], c="w", va="top", ha="left", fontsize=9, transform="axes")

axes.format(
    grid=False,
    xlabel=r'$\Delta$RA (")',
    ylabel=r'$\Delta$DEC (")',
)
axes[:, 1].format(ytickloc="none")
axes[0, :].format(xtickloc="none")


bar_width_arc = 0.04
# scale bar
rect = patches.Rectangle([-0.63, -0.43], -bar_width_arc, 3e-3, color="white")
axes[2].add_patch(rect)
axes[2].text(
    -0.63 - bar_width_arc / 2,
    -0.43 + 0.01,
    f'{bar_width_arc*1e3:.0f} mas',
    c="white",
    ha="center",
    fontsize=8,
)

# compass rose
arrow_length = 0.02
delta = np.array((0, arrow_length))
axes[1, 1].plot((-0.82, delta[0] + -0.82), (-0.43, delta[1] + -0.43), color="w", lw=1)
axes[1, 1].text(
    delta[0] - 0.82,
    -0.43 + delta[1],
    "N",
    color="w",
    fontsize=7,
    ha="center",
    va="bottom",
)
delta = np.array((arrow_length, 0))
axes[1, 1].plot((-0.82, delta[0] + -0.82), (-0.43, delta[1] + -0.43), color="w", lw=1)
axes[1, 1].text(
    delta[0] - 0.82 + 0.002,
    -0.43 + delta[1],
    "E",
    color="w",
    fontsize=7,
    ha="right",
    va="center",
)

# save output
# pro.show()
fig.savefig(paths.figures / "20230711_HD1160_mosaic.pdf", dpi=300)
