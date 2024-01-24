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
    nrows=2,
    width="3.5in",
    sharex=1,
    hspace=0.25
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
Qphi_sum = np.nansum(Qphi_frames, axis=0)
I_frames = stokes_cube[:, 0]

vmin=0
# vmax=np.nanmax(stokes_cube[:, 3])
# vmax=np.nanpercentile(Qphi_frames, 99)
bar_width_au = 20
plx = 8.7053e-3  # "
bar_width_arc = bar_width_au * plx  # "

side_length = Qphi_frames.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)
titles = ("F610", "F670", "F720", "F760")

im = axes[0].imshow(Qphi_sum, extent=ext, vmin=0)
rect = patches.Rectangle([-0.55, -0.495], bar_width_arc, 1e-2, color="white")
axes[0].add_patch(rect)
axes[0].text(-0.55 + bar_width_arc/2, -0.46, f"{bar_width_arc:.02f}\"", c="white", ha="center", fontsize=8)
axes[0].text(-0.55 + bar_width_arc/2, -0.55, f"{bar_width_au:.0f} au", c="white", ha="center", fontsize=8)
axes[0].text(0.03, 0.92, r"Stokes $Q_\phi$", transform="axes", c="white", fontsize=11)
axes[1].text(0.03, 0.92, r"Stokes $Q_\phi \times r^2$", transform="axes", c="white", fontsize=11, bbox=dict(fc="k", alpha=0.6))

vmax=np.nanmax(Qphi_sum) * (0.2)**2
im = axes[1].imshow(Qphi_sum * rs, extent=ext, vmin=0, vmax=vmax)


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
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
axes[0].format(xticks=[])

fig.savefig(
    paths.figures / "20230707_HD169142_Qphi_mosaic.pdf",
    dpi=300,
)
### FLUX plots
fig, axes = pro.subplots(nrows=2, width="3.5in", height="3.5in", sharey=0, hspace=0)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, 4))
pxscale = header["PXSCALE"] / 1e3
fwhm = 4
radii = np.arange(0.105 / pxscale - fwhm/2, 1.4 / pxscale, fwhm)
for i in range(4):
    Qphi_prof = RadialProfile(Qphi_frames[i], (center[1], center[0]), radii)
    I_prof = RadialProfile(I_frames[i], (center[1], center[0]), radii)

    common = dict(ms=2, c=cycle[i], zorder=100+i)
    axes[0].plot(Qphi_prof.radius * pxscale, Qphi_prof.profile, m="o", label=titles[i], **common)
    axes[1].plot(Qphi_prof.radius * pxscale, Qphi_prof.profile / I_prof.profile * 100, m="s", **common)

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
