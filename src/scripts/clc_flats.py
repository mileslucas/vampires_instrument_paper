import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from astropy.nddata import Cutout2D

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "mono_r"

masks = (
    "CLC-2",
    "CLC-3",
    "CLC-5",
    "CLC-7"
)

data_dict = {}
for path in (paths.data / "mask_flats").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    key = hdr["U_FLDSTP"]
    data_dict[key] = np.nan_to_num(np.squeeze(data))


plate_scale = 6.03 # mas / px

fig, axes = pro.subplots(
    nrows=2,
    ncols=2,
    width="3.5in",
    space=0
)

centers = {
    "CLC-2": (310.2, 207.1),
    "CLC-3": (321.2, 241.8),
    "CLC-5": (321.2, 241.3),
    "CLC-7": (310.9, 205.3),
}


for ax, key in zip(axes, centers):
    frame = data_dict[key]
    cutout = Cutout2D(frame, centers[key], int(800/plate_scale))
    side_length = np.array(cutout.shape) * plate_scale / 1e3
    ext = (-side_length[1]/2, side_length[1]/2, -side_length[0]/2, side_length[0]/2)
    ax.imshow(cutout.data, extent=ext)
    ax.text(0.03, 0.97, key, c="k", va="top", ha="left", fontsize=7, transform="axes")

axes.format(
    grid=False,
    xlabel=r'$\Delta x$ (")',
    ylabel=r'$\Delta y$ (")',
)
axes[:, 1].format(ytickloc="none")

# save output
fig.savefig(paths.figures / "clc_masks.pdf", dpi=300)
