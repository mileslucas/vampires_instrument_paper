import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from astropy.nddata import Cutout2D

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"

names = ("ADI only", "ADI+SDI")

adi_frame, header = fits.getdata(paths.data / "20230711_HD1160" / "20230711_HD1160_GreeDS20_all.fits", header=True)
adisdi_frame, header = fits.getdata(paths.data / "20230711_HD1160" / "20230711_HD1160_GreeDS20_ASDI.fits", header=True)

plate_scale = 5.9  # mas / px

fig, axes = pro.subplots(ncols=2, width="3.5in", space=0)

exp_sep = 135.18
exp_pa = 245.1
cy, cx = np.array(adi_frame.shape[-2:]) / 2 - 0.5
dx = -exp_sep * np.sin(np.radians(exp_pa))
dy = exp_sep * np.cos(np.radians(exp_pa))


cutout = Cutout2D(adi_frame, (cx + dx, cy + dy), 40)
yext, xext = cutout.bbox_original
ext = (
    (cx - xext[0]) * plate_scale / 1e3,
    (cx - xext[1]) * plate_scale / 1e3,
    (yext[0] - cy) * plate_scale / 1e3,
    (yext[1] - cy) * plate_scale / 1e3
)
axes[0].imshow(cutout.data, extent=ext, cmap="magma", origin="lower")
title = axes[0].text(0.05, 0.95, names[0], c="w", va="top", ha="left", fontsize=9, transform="axes")

cutout = Cutout2D(adisdi_frame, (cx + dx, cy + dy), 40)
axes[1].imshow(cutout.data, extent=ext, cmap="magma", origin="lower")
title = axes[1].text(0.05, 0.95, names[1], c="w", va="top", ha="left", fontsize=9, transform="axes")


axes.format(
    grid=False,
    xlabel=r'$\Delta$RA (")',
    ylabel=r'$\Delta$DEC (")',
)
axes[1].format(ytickloc="none")

# save output
fig.savefig(paths.figures / "HD1160_sdi_mosaic.pdf", dpi=300)
