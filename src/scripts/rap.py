import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from photutils.profiles import RadialProfile
import utils_strehl as strehl
from scipy.signal import savgol_filter

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "mono_r"
pro.rc["cycle"] = "ggplot"

mask_names = {"RAP": "Pupil (RAP)", "PSF": "PSF (Open)"}

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

plate_scale = 5.9  # mas / px

fig, axes = pro.subplots(
    [[1, 2], [3, 3]],
    wspace=0.25,
    hspace=0.5,
    width="3.5in",
    share=0,
    hratios=(0.6, 0.4),
)

for ax, key in zip(axes[0, :], mask_names):
    frame = data_dict[key]
    cy, cx = np.array(frame.shape[-2:]) / 2 - 0.5
    if key == "RAP":
        c = "w"
        cmap = "mono_r"
        window = 800
        cutout = Cutout2D(frame, (cx, cy - 3), window)
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

axes[0, 1].line(
    (0.14e3 / plate_scale, 0.78e3 / plate_scale), (0, 0), c="w", ls="--", lw=0.75
)
axes[0, 1].text(
    0.14e3 / plate_scale + (0.78e3 - 0.14e3) / 2 / plate_scale,
    0 + 3,
    '0.1" - 0.8"',
    c="w",
    fontsize=6,
    ha="center",
    va="bottom",
)

axes[0, :].format(grid=False, xticks=[], yticks=[])


ideal_psf = strehl.create_synth_psf("Open", npix=401)

radii = np.arange(0, 1.2e3 / plate_scale)
cy, cx = np.array(cutout.data.shape) / 2 - 0.5
prof = RadialProfile(cutout.data, (cx, cy), radii)
prof.normalize()

cy, cx = np.array(ideal_psf.shape) / 2 - 0.5
prof_ideal = RadialProfile(ideal_psf, (cx, cy), radii)
prof_ideal.normalize()
axes[1, :].plot(prof.radius * plate_scale / 1e3, prof.profile, label="RAP", zorder=999)
axes[1, :].plot(prof_ideal.radius * plate_scale / 1e3, prof_ideal.profile, label="PSF", c="C3")


axes[1, :].legend()
axes[1, :].format(
    yscale="log", yformatter="log", xlabel='separation (")', ylabel="norm. profile"
)

# save output
fig.savefig(paths.figures / "rap.pdf", dpi=300)
