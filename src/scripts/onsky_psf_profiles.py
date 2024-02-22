import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from photutils.profiles import RadialProfile, CurveOfGrowth
from skimage import filters

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


def get_profiles(frame):
    ny = frame.shape[-2]
    nx = frame.shape[-1]
    center = (ny - 1) / 2, (nx - 1) / 2

    rs = np.linspace(0, 170, 1000)
    radprof = RadialProfile(frame, center[::-1], rs)
    radprof.normalize()
    cog = CurveOfGrowth(frame, center[::-1], rs[1:])
    cog.normalize()
    return radprof, cog


ideal_psf, hdr = fits.getdata(paths.data / "VAMPIRES_F720_synthpsf.fits", header=True)
radprof_ideal, cog_ideal = get_profiles(ideal_psf)
ideal_norm = np.nansum(radprof_ideal.profile)


mbi_cube, hdr_good = fits.getdata(
    paths.data / "20230707_HD191195_frame.fits", header=True
)
radprof_good, cog_good = get_profiles(mbi_cube[2])
good_norm = np.nansum(radprof_good.profile)
print(f"Good FWHM: {radprof_good.gaussian_fwhm * hdr_good['PXSCALE']:.01f} mas")
print(f"Good Strehl est: {measure_strehl_otf(mbi_cube[2], ideal_psf)*100:.01f}%")

mbi_cube, hdr_fuzzy = fits.getdata(
    paths.data / "20230627_BD332642_frame.fits", header=True
)
radprof_fuzzy, cog_fuzzy = get_profiles(filters.gaussian(mbi_cube[2] + 11, 4))
fuzzy_norm = np.nansum(radprof_fuzzy.profile)
print(
    f"Long exposure FWHM: {radprof_fuzzy.gaussian_fwhm * hdr_fuzzy['PXSCALE']:.01f} mas"
)
print(
    f"Long exposure Strehl est: {measure_strehl_otf(mbi_cube[2], ideal_psf)*100:.01f}%"
)

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
    radprof_good.radii[:-1] * hdr_good["PXSCALE"] / 1e3,
    radprof_good.profile * ideal_norm / good_norm,
    c="C0",
    lw=1.25,
    label="PSF",
)
axes[0].plot(
    radprof_fuzzy.radii[:-1] * hdr_fuzzy["PXSCALE"] / 1e3,
    radprof_fuzzy.profile * ideal_norm / fuzzy_norm,
    c="C0",
    ls="--",
    lw=1.25,
    label="Long exposure",
)
## EE plots
axes[1].plot(
    [0, *cog_ideal.radii * 6.5e-3],
    [0, *cog_ideal.profile],
    c="C3",
    lw=1.25,
    label="Ideal",
)
axes[1].plot(
    [0, *cog_good.radii * hdr_good["PXSCALE"] / 1e3],
    [0, *cog_good.profile],
    c="C0",
    lw=1,
    label="PSF",
)
axes[1].plot(
    [0, *cog_fuzzy.radii * hdr_fuzzy["PXSCALE"] / 1e3],
    [0, *cog_fuzzy.profile],
    c="C0",
    ls="--",
    lw=1,
    label="Long exposure",
)

ave_wave = 720e-9
ave_lamd = np.rad2deg(ave_wave / 7.95) * 3.6e3

for ax in axes:
    ax.dualx(lambda x: x / ave_lamd, label=r"separation ($\lambda/D$)")
    ax.axvline(46.6 / 2 * ave_lamd, c="0.5", ls=":", lw=1, zorder=0)
axes[0].text(
    46.6 / 2 * ave_lamd + 1.5e-2,
    0.01,
    "SCExAO\ncontrol radius",
    c="0.5",
    fontsize=8,
    va="center",
    ha="left",
)

axes[0].legend(ncols=1)
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
fig.savefig(paths.figures / "onsky_psf_profiles.pdf", dpi=300)
