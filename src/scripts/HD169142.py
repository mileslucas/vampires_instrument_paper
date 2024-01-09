import paths
import proplot as pro
import numpy as np
from astropy.io import fits
from matplotlib import patches

pro.rc["legend.fontsize"] = 7
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"


fig, axes = pro.subplots(
    ncols=4,
    nrows=2,
    width="7in",
    # height="2.5in",
    space=0
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

vmin=0
# vmax=np.nanmax(stokes_cube[:, 3])
# vmax=np.nanpercentile(Qphi_frames, 99)


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
    # if ratio > 1:
    #     ax.text(-0.45, 0.4, f"{ratio:.0f}x", c="white", ha="right", fontsize=8)
for i in range(4):
    ax = axes[1, i]
    frame = Qphi_frames[i] * rs
    title = titles[i]
    # ratio = np.round(vmax / np.nanmax(frame))
    vmax=np.nanmax(Qphi_frames[i]) * (0.2)**2
    # vmax=None
    im = ax.imshow(frame, extent=ext, vmin=0, vmax=vmax)
    # ax.colorbar(im, loc="top")

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


# fig.colorbar(im, loc="r", label=r"mJy / sq. arcsec")

## sup title
axes.format(
    xlim=(0.6, -0.6),
    ylim=(-0.6, 0.6),
    # suptitle=f"2023/07/07 VAMPIRES HD 169142"
)
axes.format(
    xlabel='$\Delta$RA (")',
    ylabel='$\Delta$DEC (")',
)
# for ax in axes:
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
# axes[1:].format(xticks=[], yticks=[], ylabel="")
fig.savefig(
    paths.figures / f"20230707_HD169142_Qphi_mosaic.pdf",
    dpi=300,
)