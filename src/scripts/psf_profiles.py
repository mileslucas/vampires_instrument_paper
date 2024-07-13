import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from photutils.profiles import RadialProfile, CurveOfGrowth
from skimage import filters
import utils_strehl as strehl
import utils_psf_fitting as psf_fitting

pro.rc["legend.fontsize"] = 8
pro.rc["legend.frameon"] = True
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["cycle"] = "ggplot"


def measure_strehl_otf(image, psf):
    im_mtf = np.abs(np.fft.fft2(np.nan_to_num(image)))
    psf_mtf = np.abs(np.fft.fft2(np.nan_to_num(psf)))
    im_norm = im_mtf / np.max(im_mtf)
    psf_norm = psf_mtf / np.max(psf_mtf)
    return np.mean(im_norm) / np.mean(psf_norm)


def get_profiles(frame, error=None, norm=True):
    ny = frame.shape[-2]
    nx = frame.shape[-1]
    center = (ny - 1) / 2, (nx - 1) / 2

    rs = np.linspace(0, 170, 1000)
    radprof = RadialProfile(frame, center[::-1], rs, error=error)
    cog = CurveOfGrowth(frame, center[::-1], rs[1:], error=error)
    if norm:
        radprof.normalize()
        cog.normalize()
    return radprof, cog


ideal_psf = strehl.create_synth_psf("F720", npix=401)
radprof_ideal, cog_ideal = get_profiles(ideal_psf)
ideal_norm = np.nansum(radprof_ideal.profile)


ideal_strehl = strehl.measure_strehl(ideal_psf, ideal_psf)
print(f"ideal strehl: {ideal_strehl*100:.01f}%")
with fits.open(paths.data / "20230707_HD191195_frame.fits") as hdul:
    mbi_cube = hdul[0].data
    mbi_error = np.sqrt(mbi_cube)
    hdr_good = hdul[0].header
radprof_good, cog_good = get_profiles(mbi_cube[2], mbi_error[2])
good_norm = np.nansum(radprof_good.profile)
good_fit = psf_fitting.fit_moffat(mbi_cube[2])
print(f"Good FWHM: {radprof_good.gaussian_fwhm * 5.9:.01f} mas")
print(f"Good FWHM: {np.sqrt(0.5 *(good_fit.fwhmx**2 + good_fit.fwhmy**2)) * 5.9:.01f} mas")

good_strehl = strehl.measure_strehl(mbi_cube[2], ideal_psf)
print(f"Good Strehl est: {good_strehl*100:.01f}%")

mbi_cube, hdr_fuzzy = fits.getdata(
    paths.data / "20230627_BD332642_frame.fits", header=True
)
radprof_fuzzy, cog_fuzzy = get_profiles(filters.gaussian(mbi_cube[2] + 11, 4))
fuzzy_norm = np.nansum(radprof_fuzzy.profile)
fuzzy_fit = psf_fitting.fit_moffat(mbi_cube[2])
print(
    f"Bad FWHM: {radprof_fuzzy.gaussian_fwhm * 5.9:.01f} mas"
)
print(f"Bad FWHM: {np.sqrt(0.5 *(fuzzy_fit.fwhmx**2 + fuzzy_fit.fwhmy**2)) * 5.9:.01f} mas")
fuzzy_strehl = strehl.measure_strehl(mbi_cube[2], ideal_psf)
print(
    f"Bad Strehl est: {fuzzy_strehl*100:.01f}%"
)



# with fits.open(paths.data / "HD102438_adi_cube.fits") as hdul:
#     clc5_frame = hdul[0].data[159, 2]
#     clc5_header = hdul[0].header
#     X = 0.05 / 721e-3
#     clc5_factor = 6.517 * X**2 - 0.048 * X

# radprof_clc5, cog_clc5 = get_profiles(clc5_frame)
# clc5_norm = np.nansum(radprof_clc5.profile)


with fits.open(paths.data / "20230711_HD1160" / "HD1160_adi_cube.fits") as hdul:
    clc3_frame = hdul[0].data[39, 2]
    clc3_err = np.sqrt(clc3_frame)
    clc3_header = hdul[0].header
    X = 0.025 / 721e-3
    clc3_factor = 6.517 * X**2 - 0.048 * X

radprof_clc3, cog_clc3 = get_profiles(clc3_frame, clc3_err)
clc3_norm = np.nansum(radprof_clc3.profile)

fig, axes = pro.subplots(
    ncols=2, width="7in", height="3in", wratios=(0.65, 0.35), wspace=1.5, share=0
)

axes[0].plot(
    radprof_ideal.radii[:-1] * 5.9e-3,
    radprof_ideal.profile,
    c="C3",
    lw=1.25,
    label="Ideal",
)
axes[0].plot(
    radprof_good.radii[:-1] * 5.9 / 1e3,
    radprof_good.profile * ideal_norm / good_norm,
    c="C0",
    lw=1.25,
    label="PSF",
)
axes[0].plot(
    radprof_good.radii[:-1] * 5.9 / 1e3,
    radprof_good.profile_error * ideal_norm / good_norm,
    c="C0",
    ls=":",
    lw=1.25,
    label=r"$1\sigma$ PSF",
)
axes[0].plot(
    radprof_fuzzy.radii[:-1] * 5.9 / 1e3,
    radprof_fuzzy.profile * ideal_norm / fuzzy_norm,
    c="C0",
    ls="--",
    lw=1.25,
    label="Bad seeing",
)
clc3_mask = radprof_clc3.radii[:-1] * 5.9 / 1e3 > 59e-3
axes[0].plot(
    radprof_clc3.radii[:-1][clc3_mask] * 5.9 / 1e3,
    radprof_clc3.profile[clc3_mask] * ideal_norm / clc3_norm * clc3_factor,
    c="C1",
    lw=1.25,
    label="Coron.",
)
axes[0].plot(
    radprof_clc3.radii[:-1][clc3_mask] * 5.9 / 1e3,
    radprof_clc3.profile_error[clc3_mask] * ideal_norm / clc3_norm * clc3_factor,
    c="C1",
    ls=":",
    lw=1.25,
    label=r"$1\sigma$ Coron.",
)
# clc5_mask = radprof_clc5.radii[:-1] * 5.9 / 1e3 > 105e-3
# axes[0].plot(
#     radprof_clc5.radii[:-1][clc5_mask] * 5.9 / 1e3,
#     radprof_clc5.profile[clc5_mask] * ideal_norm / clc5_norm * clc5_factor,
#     c="C1",
#     ls=":",
#     lw=1.25,
#     label="Coron. (CLC-5)",
# )
## EE plots
axes[1].plot(
    [0, *cog_ideal.radii * 5.9e-3],
    [0, *cog_ideal.profile],
    c="C3",
    lw=1.25,
    label="Ideal",
)
axes[1].plot(
    [0, *cog_good.radii * 5.9e-3],
    [0, *cog_good.profile],
    c="C0",
    lw=1,
    label="PSF",
)
axes[1].plot(
    [0, *cog_fuzzy.radii * 5.9e-3],
    [0, *cog_fuzzy.profile],
    c="C0",
    ls="--",
    lw=1,
    label="Bad seeing",
)

ave_wave = 721e-9
ave_lamd = np.rad2deg(ave_wave / 7.95) * 3.6e3

for ax in axes:
    ax.dualx(lambda x: x / ave_lamd, label=r"separation ($\lambda/D$)")
    ax.axvline(46.6 / 2 * ave_lamd, c="0.5", lw=1, zorder=0)

ymin, _ = axes[0].get_ylim()
axes[0].text(
    46.6 / 2 * ave_lamd - 1.5e-2,
    0.03,
    "SCExAO\ncontrol radius",
    c="0.3",
    fontsize=8,
    rotation=90,
    va="center",
    ha="right",
)

axes[0].legend(ncols=1, frame=False)
# axes[1].legend(ncols=2)
axes.format(
    xlim=(None, 0.8),
    grid=True,
    xlabel='separation (")',
)
axes[1].format(xlim=(None, 0.6))
axes[0].format(
    ylabel="normalized profile",
    yscale="log",
    yformatter="log",
    # xtickloc="top",
)
axes[1].format(ylabel="encircled energy", ylim=(0, None), ytickloc="right")
# save output
fig.savefig(paths.figures / "psf_profiles.pdf", dpi=300)
