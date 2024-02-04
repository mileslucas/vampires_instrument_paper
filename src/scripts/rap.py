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

mask_names = {"RAP": "Pupil (RAP)", "PSF": "PSF"}

data_dict = {}
for path in [
    *list((paths.data).glob("coro_images/*.fits")),
    *list((paths.data).glob("pupil_images/*.fits")),
]:
    data, hdr = fits.getdata(path, header=True)
    if hdr["U_FLDSTP"] != "Fieldstop" or hdr["U_MASK"] != "RAP":
        continue
    if hdr["U_PUPST"] == "IN":
        key = "RAP"
    else:
        key = "PSF"
    data_dict[key] = np.nan_to_num(np.squeeze(data))


plate_scale = 6.03  # mas / px

fig, axes = pro.subplots(ncols=2, wspace=0.25, width="3.5in", share=0)

for ax, key in zip(axes, mask_names):
    frame = data_dict[key]
    cy, cx = np.array(frame.shape[-2:]) / 2 - 0.5
    if key == "RAP":
        c = "w"
        cmap = "mono_r"
        window = 800
        cutout = Cutout2D(frame, (cx, cy), window)
        norm = simple_norm(cutout.data, "asinh")
        vmax = None
    else:
        c = "k"
        vmax = np.nanpercentile(cutout.data, 30)
        norm = None
        cmap = "magma"
        window = int(2e3 / 6.03)
        cutout = Cutout2D(frame, (cx, cy), window)
    side_length = np.array(cutout.shape)
    ext = (
        -side_length[1] / 2,
        side_length[1] / 2,
        -side_length[0] / 2,
        side_length[0] / 2,
    )
    ax.imshow(cutout.data, extent=ext, cmap=cmap, norm=norm, vmin=0, vmax=vmax)
    ax.text(
        0.03,
        0.97,
        mask_names[key],
        c="w",
        va="top",
        ha="left",
        fontsize=7,
        transform="axes",
    )

axes[1].line((0.14e3 / plate_scale, 0.78e3 / plate_scale), (0, 0), c="w", ls="--", lw=0.75)
axes[1].text(
    0.14e3 / plate_scale + (0.78e3 - 0.14e3) / 2 / plate_scale,
    0 + 3,
    '0.1" - 0.8"',
    c="w",
    fontsize=6,
    ha="center",
    va="bottom",
)

axes.format(grid=False, xticks=[], yticks=[])

# save output
fig.savefig(paths.figures / "rap.pdf", dpi=300)
