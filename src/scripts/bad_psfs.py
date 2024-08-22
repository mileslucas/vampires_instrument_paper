import paths
import proplot as pro
import numpy as np
from astropy.io import fits
from matplotlib import patches
from astropy.visualization import simple_norm
from scipy.ndimage import shift

pro.rc["legend.fontsize"] = 8
pro.rc["font.size"] = 9
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "magma"
pro.rc["axes.grid"] = False

lwe_cube, lwe_header = fits.getdata(
    paths.data / "20230831_HR718_calib_LWE.fits", header=True
)
lwe_frame = shift(
    lwe_cube[127],
    (np.array(lwe_cube.shape[-2:]) / 2 - 0.5) - np.array((283.404, 262.763)),
    order=5,
)

mbi_cube, hdr_fuzzy = fits.getdata(
    paths.data / "20230627_BD332642_frame.fits", header=True
)
long_expo_frame = mbi_cube[2] + 11

good_cube, good_hdr = fits.getdata(
    paths.data / "20230707_HD191195_frame.fits", header=True
)
good_frame = good_cube[2]

fig, axes = pro.subplots(ncols=3, wspace=0.25, width="3.5in")


side_length = good_frame.shape[-1] * 5.9 * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)
im = axes[0].imshow(
    good_frame,
    extent=ext,
    norm=simple_norm(good_frame, stretch="sqrt"),
    vmin=0,
)

side_length = lwe_frame.shape[-1] * 5.9 * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)
im = axes[1].imshow(
    lwe_frame,
    extent=ext,
    norm=simple_norm(lwe_frame, stretch="sqrt"),
    vmin=0,
)

side_length = long_expo_frame.shape[-1] * 5.9 * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)
im = axes[2].imshow(
    long_expo_frame,
    extent=ext,
    norm=simple_norm(long_expo_frame, stretch="sqrt"),
    vmin=0,
)

axes[0].text(0.05, 0.87, "Good PSF", transform="axes", c="white", ha="left", fontsize=8)
axes[1].text(0.05, 0.87, "LWE", transform="axes", c="white", ha="left", fontsize=8)
axes[2].text(
    0.05, 0.87, "Bad seeing", transform="axes", c="white", ha="left", fontsize=8
)

axes[0].text(
    0.95,
    0.05,
    f"DIT={good_hdr['EXPTIME']*1e3:.01f} ms",
    transform="axes",
    c="white",
    ha="right",
    fontsize=7,
)
axes[1].text(
    0.95,
    0.05,
    f"DIT={lwe_header['EXPTIME']*1e3:.01f} ms",
    transform="axes",
    c="white",
    ha="right",
    fontsize=7,
)
axes[2].text(
    0.95,
    0.05,
    f"DIT={hdr_fuzzy['EXPTIME']:.01f} s",
    transform="axes",
    c="white",
    ha="right",
    fontsize=7,
)

bar_width_arc = 0.1

## scale bar
rect = patches.Rectangle([0.17, -0.175], -bar_width_arc, 3e-3, color="white")
axes[0].add_patch(rect)
axes[0].text(
    0.17 - bar_width_arc / 2,
    -0.158,
    f'{bar_width_arc:.01f}"',
    c="white",
    ha="center",
    fontsize=7,
)


## sup title
axes.format(
    xlim=(0.2, -0.2),
    ylim=(-0.2, 0.2),
)
axes.format(xtickloc="none", ytickloc="none")

fig.savefig(
    paths.figures / "bad_psf_mosaic.pdf",
    dpi=300,
)
