import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "mono_r"

masks_name = {
    "Open": "Pupil",
    "PupilRef": "PupilRef",
    "LyotOpt": "LyotStop-M",
    "LyotStop": "LyotStop-L",
}

data_dict = {}
for path in (paths.data / "pupil_images").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    if hdr["U_FLDSTP"] != "Fieldstop":
        continue
    key = hdr["U_MASK"]
    data_dict[key] = np.nan_to_num(np.squeeze(data))


# plate_scale = 6.03 # mas / px

fig, axes = pro.subplots(nrows=2, ncols=2, width="3.5in", space=0)

for ax, key in zip(axes, masks_name):
    frame = data_dict[key]
    cy, cx = np.array(frame.shape[-2:]) / 2 - 0.5
    cutout = Cutout2D(frame, (cx, cy), 850)
    side_length = np.array(cutout.shape)
    ext = (
        -side_length[1] / 2,
        side_length[1] / 2,
        -side_length[0] / 2,
        side_length[0] / 2,
    )
    ax.imshow(
        cutout.data, extent=ext, cmap="mono_r", norm=simple_norm(cutout.data, "asinh")
    )
    ax.text(
        0.03,
        0.97,
        masks_name[key],
        c="w",
        va="top",
        ha="left",
        fontsize=7,
        transform="axes",
    )

axes.format(grid=False, xticks=[], yticks=[])
axes[:, 1].format(ytickloc="none")

# save output
fig.savefig(paths.figures / "pupil_masks.pdf", dpi=300)
