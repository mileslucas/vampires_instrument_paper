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

masks = ("DGVVC (PBS)", "DGVVC (NPBS)")

data_dict = {}
for path in (paths.data / "mask_flats").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    if hdr["U_FLDSTP"] != "DGVVC":
        continue
    key = f'{hdr["U_FLDSTP"]} ({hdr["U_BS"]})'
    data_dict[key] = np.nan_to_num(np.squeeze(data))


plate_scale = 6.03  # mas / px

fig, axes = pro.subplots(
    ncols=2,
    wspace=0,
    width="3.5in",
)


for ax, key in zip(axes, masks):
    frame = data_dict[key]
    cutout = Cutout2D(frame, (268.5, 268.5), int(2e3 / plate_scale), mode="partial")
    side_length = np.array(cutout.shape) * plate_scale / 1e3
    ext = (
        -side_length[1] / 2,
        side_length[1] / 2,
        -side_length[0] / 2,
        side_length[0] / 2,
    )
    ax.imshow(cutout.data, vmin=0.85, vmax=1.15, extent=ext)
    text_mask, text_bs = key.split()
    ax.text(
        0.03,
        1.03,
        text_mask,
        c="k",
        va="bottom",
        ha="left",
        fontsize=7,
        transform="axes",
    )
    ax.text(
        0.97,
        1.03,
        text_bs,
        c="k",
        va="bottom",
        ha="right",
        fontsize=7,
        transform="axes",
    )
axes[:, 1].format(ytickloc="none")

axes.format(
    grid=False,
    xlabel=r'$\Delta x$ (")',
    ylabel=r'$\Delta y$ (")',
)

# save output
fig.savefig(paths.figures / "dgvvc_masks.pdf", dpi=300)
