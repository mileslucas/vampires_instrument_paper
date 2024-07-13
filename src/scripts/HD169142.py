import paths
import proplot as pro
import numpy as np
from astropy.io import fits
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from photutils.profiles import RadialProfile
from scipy.optimize import minimize_scalar

pro.rc["legend.fontsize"] = 7
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"
pro.rc["axes.grid"] = False


stokes_path = paths.data / "20230707_HD169142_vampires_stokes_cube.fits"
stokes_cube, header = fits.getdata(stokes_path, header=True)

plate_scale = 5.9  # mas / px

ny = stokes_cube.shape[-2]
nx = stokes_cube.shape[-1]
center = (ny - 1) / 2, (nx - 1) / 2
Ys, Xs = np.ogrid[: stokes_cube.shape[-2], : stokes_cube.shape[-1]]

radii = np.hypot(Ys - center[-2], Xs - center[-1])
angles = np.arctan2(center[-1] - Xs, Ys - center[-2])


def azimuthal_stokes(Q, U, phi=0):
    cos2th = np.cos(2 * (angles + phi))
    sin2th = np.sin(2 * (angles + phi))
    Qphi = -Q * cos2th - U * sin2th
    Uphi = Q * sin2th - U * cos2th
    return Qphi, Uphi


def opt_func(phi, stokes_frame):
    Qr, Ur = azimuthal_stokes(stokes_frame[2], stokes_frame[3], phi)
    return np.nanmean(Ur**2)


def optimize_Uphi(stokes_frames, frame=""):
    res = minimize_scalar(
        lambda f: opt_func(f, stokes_frames), bounds=(-np.pi / 4, np.pi / 4)
    )
    print(f"HD169142 {frame} field phi offset: {np.rad2deg(res.x):.01f}°")
    return azimuthal_stokes(stokes_frames[2], stokes_frames[3], phi=res.x)


titles = ("F610", "F670", "F720", "F760")

for i in range(4):
    Qphi, Uphi = optimize_Uphi(stokes_cube[i], titles[i])
    stokes_cube[i, 4] = Qphi
    stokes_cube[i, 5] = Uphi

rs = (radii * plate_scale / 1e3) ** 2

Qphi_frames = stokes_cube[:, 4]
Qphi_sum = np.nansum(Qphi_frames, axis=0)
I_frames = 0.5 * (stokes_cube[:, 0] + stokes_cube[:, 1])

vmin = 0
bar_width_au = 20
plx = 8.7053e-3  # "
bar_width_arc = bar_width_au * plx  # "

side_length = Qphi_frames.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

### Mosaic plots

fig, axes = pro.subplots([[1, 2, 5], [3, 4, 6]], width="7in", hspace=0.25, wspace=[0.25, 0.75], spanx=False)

for Qphi, ax, title in zip(Qphi_frames, axes, titles):
    im = ax.imshow(Qphi, extent=ext, vmin=0, vmax=0.9 * np.nanmax(Qphi))
    ax.text(
        0.03, 0.97, title, transform="axes", c="white", ha="left", va="top", fontsize=9
    )

im = axes[4].imshow(Qphi_sum, extent=ext, vmin=0, vmax=0.9 * np.nanmax(Qphi_sum))
axes[4].text(0.03, 0.92, r"Mean", transform="axes", c="white", fontsize=9)

vmax = np.nanmax(Qphi_sum) * (0.2) ** 2
im = axes[5].imshow(Qphi_sum * rs, extent=ext, vmin=0, vmax=vmax)
axes[5].text(
    0.03,
    0.92,
    r"Mean$\times r^2$",
    transform="axes",
    c="white",
    fontsize=9,
    bbox=dict(fc="k", alpha=0.6),
)

for ax in axes:
    # coronagraph mask
    ax.scatter(
        0,
        0,
        color="white",
        alpha=0.8,
        marker="+",
        ms=20,
        lw=0.5,
        zorder=999,
    )
    circ = patches.Circle([0, 0], 105e-3, ec="white", fc="k", lw=1)
    ax.add_patch(circ)
# scale bar
rect = patches.Rectangle([0.55, -0.485], -bar_width_arc, 8e-3, color="white")
axes[2].add_patch(rect)
axes[2].text(
    0.55 - bar_width_arc / 2,
    -0.45,
    f'{bar_width_arc:.02f}"',
    c="white",
    ha="center",
    fontsize=8,
)
axes[2].text(
    0.55 - bar_width_arc / 2,
    -0.55,
    f"{bar_width_au:.0f} au",
    c="white",
    ha="center",
    fontsize=8,
)
# compass rose
arrow_length = 0.1
delta = np.array((0, arrow_length))
axes[1, 1].plot((-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1)
axes[1, 1].text(
    delta[0] - 0.53,
    -0.53 + delta[1],
    "N",
    color="w",
    fontsize=7,
    ha="center",
    va="bottom",
)
delta = np.array((arrow_length, 0))
axes[1, 1].plot((-0.53, delta[0] + -0.53), (-0.53, delta[1] + -0.53), color="w", lw=1)
axes[1, 1].text(
    delta[0] - 0.525,
    -0.535 + delta[1],
    "E",
    color="w",
    fontsize=7,
    ha="right",
    va="center",
)


## sup title
axes.format(
    xlim=(0.6, -0.6),
    ylim=(-0.6, 0.6),
    grid=False,
    xlabel=r'$\Delta$RA (")',
    ylabel=r'$\Delta$DEC (")',
    ylocator=MaxNLocator(5, prune="both"),
    xlocator=MaxNLocator(5, prune="both")
)
axes[:, 1].format(ytickloc="none")
axes[0, :].format(xtickloc="none")

fig.savefig(
    paths.figures / "20230707_HD169142_Qphi_mosaic.pdf",
    dpi=300,
)

### FLUX plots
fig, axes = pro.subplots(nrows=2, width="3.5in", height="3.25in", sharey=0, space=0)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))
pxscale = 5.9 / 1e3
fwhm = 4
radii = np.arange(0.105 / pxscale - fwhm / 2, 1.4 / pxscale, fwhm)
for i in range(4):
    Qphi_prof = RadialProfile(Qphi_frames[i], (center[1], center[0]), radii)
    _mask = Qphi_prof.profile < 1e-2
    Qphi_prof.profile[_mask] = np.nan
    I_prof = RadialProfile(I_frames[i], (center[1], center[0]), radii)

    common = dict(ms=2, c=cycle[i], zorder=100 + i)
    axes[0].plot(
        Qphi_prof.radius * pxscale, Qphi_prof.profile, m="o", label=titles[i], **common
    )
    axes[1].plot(
        Qphi_prof.radius * pxscale,
        Qphi_prof.profile / I_prof.profile * 100,
        m="s",
        **common,
    )

axes[0].legend(ncols=1)
axes.format(
    xlabel='radius (")',
    grid=True,
)
axes[0].format(ylabel=r"$Q_\phi$ flux (Jy / arcsec$^2$)", yscale="log")
axes[1].format(ylabel=r"$Q_\phi/I_{tot}$ flux (%)")
fig.savefig(
    paths.figures / "20230707_HD169142_Qphi_flux.pdf",
    dpi=300,
)
