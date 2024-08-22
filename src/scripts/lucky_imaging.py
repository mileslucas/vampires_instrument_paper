import paths
from astropy.io import fits
import numpy as np
import proplot as pro
import utils_strehl as strehl
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from skimage.registration import phase_cross_correlation
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import shift

pro.rc["legend.fontsize"] = 8
pro.rc["font.size"] = 9
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["axes.grid"] = False
pro.rc["cmap"] = "magma"


def dft_centroid(frame, psf, guess=None, f=30):
    # get refined centroid first
    if guess is None:
        guess = np.unravel_index(np.nanargmax(frame), frame.shape)
    cutout = Cutout2D(
        frame, (guess[1], guess[0]), psf.shape, mode="partial", fill_value=0
    )

    dft_off, _, _ = phase_cross_correlation(
        psf, cutout.data, upsample_factor=f, normalization=None
    )
    # need to update with center of frame
    ctr_off = np.array(cutout.shape) / 2 - 0.5 - dft_off
    return np.array(
        (
            ctr_off[0] + cutout.slices_original[0].start,
            ctr_off[1] + cutout.slices_original[1].start,
        )
    )


plate_scale = 5.9  # mas/px

raw_cube = fits.getdata(paths.data / "20230831_HR718_calib_LWE.fits")
# cut off trailing frames which have a vignetting
cube = raw_cube[:1000]
ctr = np.array(cube.shape[-2:]) / 2 - 0.5
psf = strehl.create_synth_psf("750-50", npix=31)


long_exp = np.mean(cube, axis=0)
ctr_guess = np.unravel_index(long_exp.argmax(), long_exp.shape)

long_exp_ctr = dft_centroid(long_exp, psf, ctr_guess)
long_exp_shifted = shift(long_exp, ctr - long_exp_ctr)


print("Measuring centroids")
centroids = np.array([dft_centroid(frame, psf, ctr_guess) for frame in cube])
offsets = ctr[None, :] - centroids

print("Aligning cube")
registered_cube = np.array(
    [shift(frame, offset) for frame, offset in zip(cube, offsets)]
)

print("Measuring Strehl ratios")
strehls = np.array(
    [strehl.measure_strehl(frame, psf, ctr) for frame in registered_cube]
)

print("Coadding")
select_pcts = (0, 30, 60, 90)
images = []
final_strehls = [strehl.measure_strehl(long_exp_shifted, psf, ctr)]
for pct in select_pcts:
    strehl_cutoff = np.percentile(strehls, pct)
    mask = strehls >= strehl_cutoff
    frame = np.mean(registered_cube[mask], 0)
    images.append(frame)
    final_strehls.append(strehl.measure_strehl(frame, psf, ctr))

print("Plotting")
side_length = cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

## Plot mosaic
fig, axes = pro.subplots(nrows=2, ncols=5, wspace=0, hspace=0.5, width="7in")


axes[0, 0].imshow(
    long_exp_shifted,
    norm=simple_norm(long_exp_shifted, "log"),
    cmap="magma",
    extent=ext,
)
axes[0, 1].imshow(
    images[0], norm=simple_norm(images[0], "log"), cmap="magma", extent=ext
)
axes[0, 2].imshow(
    images[1], norm=simple_norm(images[1], "log"), cmap="magma", extent=ext
)
axes[0, 3].imshow(
    images[2], norm=simple_norm(images[2], "log"), cmap="magma", extent=ext
)
axes[0, 4].imshow(
    images[3], norm=simple_norm(images[3], "log"), cmap="magma", extent=ext
)

# strehls
txt_kwargs = dict(fontsize=8, c="w", transform="axes")
axes[0, 0].text(
    0.95, 0.05, f"SR={final_strehls[0]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[0, 1].text(
    0.95, 0.05, f"SR={final_strehls[1]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[0, 2].text(
    0.95, 0.05, f"SR={final_strehls[2]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[0, 3].text(
    0.95, 0.05, f"SR={final_strehls[3]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[0, 4].text(
    0.95, 0.05, f"SR={final_strehls[4]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)

#######
####### Lower S/N cube
#######
raw_cube = fits.getdata(paths.data / "20230730_HD236419_F720_cube.fits")
cube = raw_cube[:1000]
ctr = np.array(cube.shape[-2:]) / 2 - 0.5
psf = strehl.create_synth_psf("F720", npix=31)


long_exp = np.mean(cube, axis=0)
ctr_guess = np.unravel_index(long_exp.argmax(), long_exp.shape)

long_exp_ctr = dft_centroid(long_exp, psf, ctr_guess)
long_exp_shifted = shift(long_exp, ctr - long_exp_ctr)


print("Measuring centroids")
centroids = np.array([dft_centroid(frame, psf, ctr_guess) for frame in cube])
offsets = ctr[None, :] - centroids

print("Aligning cube")
registered_cube = np.array(
    [shift(frame, offset) for frame, offset in zip(cube, offsets)]
)

print("Measuring Strehl ratios")
strehls = np.array(
    [strehl.measure_strehl(frame, psf, ctr) for frame in registered_cube]
)

print("Coadding")
select_pcts = (0, 30, 60, 90)
images = []
final_strehls = [strehl.measure_strehl(long_exp_shifted, psf, ctr)]
for pct in select_pcts:
    strehl_cutoff = np.percentile(strehls, pct)
    mask = strehls >= strehl_cutoff
    frame = np.mean(registered_cube[mask], 0)
    images.append(frame)
    final_strehls.append(strehl.measure_strehl(frame, psf, ctr))

print("Plotting")
side_length = cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)


axes[1, 0].imshow(
    long_exp_shifted,
    norm=simple_norm(long_exp_shifted, "log"),
    cmap="magma",
    extent=ext,
)
axes[1, 1].imshow(
    images[0], norm=simple_norm(images[0], "log"), cmap="magma", extent=ext
)
axes[1, 2].imshow(
    images[1], norm=simple_norm(images[1], "log"), cmap="magma", extent=ext
)
axes[1, 3].imshow(
    images[2], norm=simple_norm(images[2], "log"), cmap="magma", extent=ext
)
axes[1, 4].imshow(
    images[3], norm=simple_norm(images[3], "log"), cmap="magma", extent=ext
)

# strehls
axes[1, 0].text(
    0.95, 0.05, f"SR={final_strehls[0]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[1, 1].text(
    0.95, 0.05, f"SR={final_strehls[1]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[1, 2].text(
    0.95, 0.05, f"SR={final_strehls[2]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[1, 3].text(
    0.95, 0.05, f"SR={final_strehls[3]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)
axes[1, 4].text(
    0.95, 0.05, f"SR={final_strehls[4]*100:.0f}%", ha="right", va="bottom", **txt_kwargs
)


axes.format(
    xlim=(0.25, -0.25),
    ylim=(-0.25, 0.25),
    xlabel=r'$\Delta$RA (")',
    ylabel=r'$\Delta$DEC (")',
    ylocator=MaxNLocator(3, prune="both"),
    xlocator=MaxNLocator(3, prune="both"),
    toplabels=(
        "Long Exp.",
        "Shift-and-add",
        "Discard 30%",
        "Discard 60%",
        "Discard 90%",
    ),
)
axes[0, :].format(xtickloc="none")
axes[:, 1:].format(ytickloc="none")

fig.savefig(paths.figures / "lucky_imaging_mosaic.pdf", dpi=300)
