import paths
import proplot as pro
from astropy.io import fits
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import patches
from photutils.profiles import RadialProfile
from scipy.optimize import minimize_scalar

pro.rc["cycle"] = "ggplot"
pro.rc["image.origin"] = "lower"
pro.rc["font.size"] = 9
pro.rc["legend.fontsize"] = 8

with fits.open(paths.data / "20230711_Neptune_stokes_cube.fits") as hdul:
    stokes_cube = hdul[0].data
    header = hdul[0].header
    stokes_err = hdul["ERR"].data

## data from JPL horizons
planet_diam = 2.313102  # arcsecond
np_ang = 318.0816  # deg E of N
np_dist = -1.058  # arcsec
surf_bright = 9.286  # mag / sq. arcsecond


ny = stokes_cube.shape[-2]
nx = stokes_cube.shape[-1]
center = (ny - 1) / 2, (nx - 1) / 2
Ys, Xs = np.ogrid[: stokes_cube.shape[-2], : stokes_cube.shape[-1]]
angles = np.arctan2(center[-1] - Xs, Ys - center[-2])
titles = ("F610", "F670", "F720", "F760")


def radial_stokes(Q, U, phi=0):
    cos2th = np.cos(2 * (angles + phi))
    sin2th = np.sin(2 * (angles + phi))
    Qr = Q * cos2th + U * sin2th
    Ur = -Q * sin2th + U * cos2th
    return Qr, Ur


def opt_func(phi, stokes_frame):
    Qr, Ur = radial_stokes(stokes_frame[2], stokes_frame[3], phi)
    return np.nanmean(Ur**2)


def optimize_Qr(stokes_frames, frame=""):
    res = minimize_scalar(
        lambda f: opt_func(f, stokes_frames), bounds=(-np.pi / 4, np.pi / 4)
    )
    print(f"Neptune {frame} field phi offset: {np.rad2deg(res.x):.01f}°")
    return radial_stokes(stokes_frames[2], stokes_frames[3], phi=res.x)


def mask_circle(image, x, y, rad):
    Ys, Xs = np.ogrid[: image.shape[-2], : image.shape[-1]]
    rs = np.hypot(Ys - y, Xs - x)
    mask = rs <= rad
    image[mask] = np.nan
    return image
    # return np.where(mask, np.nan, image)


def mask_all(image):
    mask_circle(image, 351.7, 326.3, 11)
    mask_circle(image, 438.7, 251.4, 8)


for i in range(4):
    Qr, Ur = optimize_Qr(stokes_cube[i], titles[i])
    stokes_cube[i, 4] = Qr
    stokes_cube[i, 5] = Ur

fig, axes = pro.subplots(
    nrows=2, ncols=4, width="7in", space=0, hspace=0.25, sharey=1, sharex=1
)

plate_scale = 5.9
side_length = stokes_cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

## Plot and save
bs_fact = 0.7964 / 2

ext_fact = np.array((0.09, 0.056, 0.040, 0.032))
extinction = ext_fact * header["AIRMASS"]
ext_lin = 10 ** (-0.4 * extinction)

Jy_fact = (
    np.array((1.2e-6, 6.1e-7, 4.4e-7, 1e-6))
    / bs_fact
    / (plate_scale**2 / 1e6)
    / ext_lin
)  # Jy / sq.arcsec / (e-/s)
calib_data = stokes_cube * Jy_fact[:, None, None, None]
Qr_frames = calib_data[:, 4]
I_frames = (calib_data[:, 0] + calib_data[:, 1]) / 2
calib_errs = stokes_err * Jy_fact[:, None, None, None]
Qr_err = calib_errs[:, 4]
I_err = np.hypot(calib_errs[:, 0], calib_errs[:, 1]) / np.sqrt(2)


side_length = stokes_cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

# PDI images
for ax, frame, title in zip(axes[0, :], Qr_frames, titles):
    im = ax.imshow(
        frame, extent=ext, cmap="bone", vmin=0, vmax=np.nanpercentile(frame, 99.5)
    )

    ax.text(0.025, 0.9, title, c="white", ha="left", fontsize=8, transform="axes")

axes[:, -1].format(
    rightlabels=(r"Stokes $Q_r$", r"Stokes $I$"),
    rightlabelpad=2,
    rightlabels_kw=dict(rotation=-90),
)

arrow_pa = np.deg2rad(np_ang - 90)
arrow_x = np_dist * np.sin(arrow_pa)
arrow_y = -np_dist * np.cos(arrow_pa)
arrow_length = 0.3


# Intensity images
mask_all(I_frames[0])
mask_all(I_frames[1])
mask_all(I_frames[2])
mask_all(I_frames[3])
for ax, frame in zip(axes[1, :], I_frames):
    im = ax.imshow(frame, extent=ext, cmap="magma", vmin=0)

for ax in axes:
    circ = patches.Circle((0, 0), planet_diam / 2, fill=False, ec="w", lw=0.5)
    ax.add_patch(circ)

    ax.arrow(
        arrow_x,
        arrow_y,
        -arrow_length * np.sin(arrow_pa),
        arrow_length * np.cos(arrow_pa),
        width=0.01,
        head_width=0,
        color="w",
        overhang=0.2,
        lw=0.1,
        length_includes_head=True,
    )

# compass rose
arrow_length = 0.2
delta = np.array((0, arrow_length))
axes[1, 3].plot((-1.15, delta[0] + -1.15), (-1.15, delta[1] + -1.15), color="w", lw=1)
axes[1, 3].text(
    delta[0] - 1.15,
    -1.15 + delta[1],
    "N",
    color="w",
    fontsize=7,
    ha="center",
    va="bottom",
)
delta = np.array((arrow_length, 0))
axes[1, 3].plot((-1.15, delta[0] + -1.15), (-1.15, delta[1] + -1.15), color="w", lw=1)
axes[1, 3].text(
    delta[0] - 1.15 + 0.02,
    -1.15 + delta[1] - 0.01,
    "E",
    color="w",
    fontsize=7,
    ha="right",
    va="center",
)


## sup title
axes.format(
    grid=False,
    xlim=(1.3, -1.3),
    ylim=(-1.3, 1.3),
    xlabel=r'$\Delta$RA (")',
    ylabel=r'$\Delta$DEC (")',
    facecolor="k",
    xlocator=0.5,
    ylocator=0.5,
)

axes[:, 1:].format(yticks=[])
axes[0, :].format(xticks=[])
fig.savefig(
    paths.figures / "20230711_Neptune_mosaic.pdf",
    dpi=300,
)


# fig, axes = pro.subplots(
#     nrows=2, ncols=4, width="7in", space=0, hspace=0.25, sharey=1, sharex=1
# )
# # PDI images
# for ax, frame, title in zip(axes[0, :], Qr_frames, titles):
#     im = ax.imshow(
#         frame, extent=ext, cmap="bone", vmin=0, vmax=np.nanpercentile(frame, 99.5)
#     )

#     ax.text(0.025, 0.9, title, c="white", ha="left", fontsize=8, transform="axes")

# axes[:, -1].format(
#     rightlabels=(r"Stokes $Q_r$", r"Stokes $I$"),
#     rightlabelpad=2,
#     rightlabels_kw=dict(rotation=-90),
# )

# arrow_pa = np.deg2rad(np_ang - 90)
# arrow_x = np_dist * np.sin(arrow_pa)
# arrow_y = -np_dist * np.cos(arrow_pa)
# arrow_length = 0.3


# # Intensity images
# for ax, frame in zip(axes[1, :], I_frames):
#     im = ax.imshow(frame, extent=ext, cmap="magma", vmin=0)

# for ax in axes:
#     circ = patches.Circle((0, 0), planet_diam / 2, fill=False, ec="w", lw=0.5)
#     ax.add_patch(circ)

#     ax.arrow(
#         arrow_x,
#         arrow_y,
#         -arrow_length * np.sin(arrow_pa),
#         arrow_length * np.cos(arrow_pa),
#         width=0.01,
#         head_width=0,
#         color="w",
#         overhang=0.2,
#         lw=0.1,
#         length_includes_head=True,
#     )

# ## sup title
# axes.format(
#     grid=False,
#     xlim=(1.3, -1.3),
#     ylim=(-1.3, 1.3),
#     xlabel=r'$\Delta$RA (")',
#     ylabel=r'$\Delta$DEC (")',
#     facecolor="k",
# )
# for ax in axes:
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

# axes[:, 1:].format(yticks=[])
# axes[0, :].format(xticks=[])

# fig.savefig(
#     paths.figures / "20230711_Neptune_mosaic.pdf",
#     dpi=300,
# )
### FLUX plots
fig, axes = pro.subplots(nrows=3, width="3.5in", height="4.5in", sharey=0, hspace=0)
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))

ny = Qr_frames.shape[-2]
nx = Qr_frames.shape[-1]
center = (ny - 1) / 2, (nx - 1) / 2

pxscale = 5.9 / 1e3
fwhm = 10
radii = np.arange(0, 1.5 / pxscale, fwhm)
for i in range(4):
    Qr_prof = RadialProfile(
        Qr_frames[i], (center[1], center[0]), radii, error=Qr_err[i]
    )
    I_prof = RadialProfile(I_frames[i], (center[1], center[0]), radii, error=I_err[i])

    common = dict(ms=3, c=cycle[i], zorder=100 + i)
    axes[0].plot(
        I_prof.radius * pxscale, I_prof.profile, m="o", label=titles[i], **common
    )
    axes[1].plot(Qr_prof.radius * pxscale, Qr_prof.profile * 1e3, m="s", **common)
    axes[2].plot(
        Qr_prof.radius * pxscale,
        Qr_prof.profile / I_prof.profile * 100,
        m="v",
        **common,
    )

[ax.axvline(planet_diam / 2, color="0.3", ls="--", lw=1) for ax in axes]
axes[2].text(
    0.82,
    0.1,
    r"$R_{planet}$",
    transform="axes",
    color="0.3",
    fontsize=7,
    fontweight="bold",
    ha="left",
)
axes[0].legend(loc="ur", ncols=1)
axes.format(
    xlabel='radius (")',
    grid=True,
)
axes[0].format(ylabel=r"$I$ flux (Jy / arcsec$^2$)")
axes[1].format(ylabel=r"$Q_r$ flux (mJy / arcsec$^2$)")
axes[2].format(ylabel=r"$Q_r/I$ flux (%)")

for ax in axes:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

fig.savefig(
    paths.figures / "20230711_Neptune_flux.pdf",
    dpi=300,
)
