import paths
import proplot as pro
import numpy as np
from astropy.io import fits
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from photutils.profiles import RadialProfile

pro.rc["legend.fontsize"] = 7
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"
pro.rc["axes.grid"] = False

fig, axes = pro.subplots(
    ncols=4,
    nrows=2,
    width="7in",
    # height="2.5in",
    space=0, hspace=0.25, sharey=1, sharex=1
)

stokes_path = paths.data / "20230707_HD169142_vampires_stokes_cube.fits"
stokes_cube, header = fits.getdata(stokes_path, header=True)

plate_scale = header["PXSCALE"] # mas / px

ny = stokes_cube.shape[-2]
nx = stokes_cube.shape[-1]
center = (ny - 1) / 2, (nx - 1) / 2
Ys, Xs = np.ogrid[: stokes_cube.shape[-2], : stokes_cube.shape[-1]]

radii = np.hypot(Ys - center[-2], Xs - center[-1])

rs = (radii * plate_scale / 1e3)**2
# rs[rs > 1] = 1

Qphi_frames = stokes_cube[:, 3]
I_frames = stokes_cube[:, 0]

vmin=0
# vmax=np.nanmax(stokes_cube[:, 3])
# vmax=np.nanpercentile(Qphi_frames, 99)
bar_width_au = 25
plx = 8.7053e-3  # "
bar_width_arc = bar_width_au * plx  # "

side_length = Qphi_frames.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)
titles = ("F610", "F670", "F720", "F760")
for i in range(4):
    ax = axes[0, i]
    frame = Qphi_frames[i]
    title = titles[i]
    # ratio = np.round(vmax / np.nanmax(frame))
    # vmax=None
    im = ax.imshow(frame, extent=ext, vmin=0)
    # ax.colorbar(im, loc="top")
    ax.text(0.025, 0.9, title, c="white", ha="left", fontsize=10, transform="axes")
    rect = patches.Rectangle([-0.5, -0.5], bar_width_arc, 1e-2, color="white")
    ax.add_patch(rect)
    ax.text(-0.5 + bar_width_arc/2, -0.45, f"{bar_width_au:.0f} au", c="white", ha="center", fontsize=7)

for i in range(4):
    ax = axes[1, i]
    frame = Qphi_frames[i] * rs
    title = titles[i]
    # ratio = np.round(vmax / np.nanmax(frame))
    vmax=np.nanmax(Qphi_frames[i]) * (0.2)**2
    # vmax=None
    im = ax.imshow(frame, extent=ext, vmin=0, vmax=vmax)
    # ax.colorbar(im, loc="top")

axes[:, -1].format(rightlabels=(r"Stokes $Q_\phi$", r"Stokes $Q_\phi \times r^2$"), rightlabelpad=2, rightlabels_kw=dict(rotation=-90))


# coronagraph mask
for ax in axes:
    ax.scatter(
        0,
        0,
        color="white",
        alpha=0.8,
        marker="+",
        ms=20,
        lw=1,
        zorder=999,
    )
    circ = patches.Circle([0, 0], 109e-3, ec="white", fc="k", lw=1)
    ax.add_patch(circ)


## sup title
axes.format(
    xlim=(0.6, -0.6),
    ylim=(-0.6, 0.6),
    # suptitle=f"2023/07/07 VAMPIRES HD 169142"
)
axes.format(
    xlabel=r'$\Delta$RA (")',
    ylabel=r'$\Delta$DEC (")',
)

for ax in axes:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

axes[:, 1:].format(yticks=[])
axes[0, :].format(xticks=[])

fig.savefig(
    paths.figures / "20230707_HD169142_Qphi_mosaic.pdf",
    dpi=300,
)
### FLUX plots
fig, axes = pro.subplots(nrows=2, width="3.5in", height="3.5in", sharey=0, hspace=0)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))
pxscale = header["PXSCALE"] / 1e3
fwhm = 4
radii = np.arange(0.105 / pxscale, 1.4 / pxscale, fwhm)
for i in range(4):
    Qphi_prof = RadialProfile(Qphi_frames[i], (center[1], center[0]), radii)
    rel_prof = RadialProfile(Qphi_frames[i] / I_frames[i], (center[1], center[0]), radii)

    common = dict(ms=2, c=cycle[i], zorder=100+i)
    axes[0].plot(Qphi_prof.radius * pxscale, Qphi_prof.profile, m="o", label=titles[i], **common)
    axes[1].plot(rel_prof.radius * pxscale, rel_prof.profile * 100, m="s", **common)

axes[0].legend(ncols=1)
axes.format(
    xlabel="radius (\")",
    grid=True,
)
axes[0].format(
    ylabel=r"$Q_\phi$ flux (Jy / arcsec$^2$)",
    yscale="log"
)
axes[1].format(
    ylabel=r"$Q_\phi/I_{tot}$ flux (%)"
)

fig.savefig(
    paths.figures / "20230707_HD169142_Qphi_flux.pdf",
    dpi=300,
)
