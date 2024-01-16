import paths
import proplot as pro
import numpy as np
from astropy.io import fits
from matplotlib import patches
from astropy.visualization import simple_norm

pro.rc["legend.fontsize"] = 7
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "matter_r"
pro.rc["axes.grid"] = False
# pro.rc["cycle"] = "ggplot"


fig, axes = pro.subplots(
    width="3.5in",
)

halpha_frame, header = fits.getdata(paths.data / "20230707_RAqr_Halpha.fits", header=True)
hacont_frame = fits.getdata(paths.data / "20230707_RAqr_Ha-cont.fits")
halpha_frame = np.flipud(halpha_frame) 
hacont_frame = np.flipud(hacont_frame) 
halpha_frame = halpha_frame / header["EXPTIME"] * header["GAIN"] * 3.6e-5
hacont_frame = hacont_frame / header["EXPTIME"] * header["GAIN"] * 1.6e-5
plate_scale = header["PXSCALE"] # mas / px
vmin=0
# vmax=np.nanmax(stokes_cube[:, 3])
# vmax=np.nanpercentile(Qphi_frames, 99)
bar_width_au = 50
plx = 2.5931e-3  # "
bar_width_arc = bar_width_au * plx # "

side_length = halpha_frame.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

im = axes[0].imshow(halpha_frame, extent=ext, norm=simple_norm(halpha_frame, stretch="log"), vmin=0)
    # # ax.colorbar(im, loc="top")
    # ax.text(0.025, 0.9, title, c="white", ha="left", fontsize=10, transform="axes")
rect = patches.Rectangle([0.35, -1], bar_width_arc, 1e-2, color="white")
axes[0].add_patch(rect)
axes[0].text(0.35 + bar_width_arc/2, -0.95, f"{bar_width_au:.0f} au", c="white", ha="center", fontsize=7)



# fig.colorbar(im, loc="r", label=r"mJy / sq. arcsec")

## sup title
axes.format(
    xlim=(0.6, -0.6),
    ylim=(-1.1, 0.85),
    # suptitle=f"2023/07/07 VAMPIRES HD 169142"
)
axes.format(
    xlabel='$\Delta$RA (")',
    ylabel='$\Delta$DEC (")',
)
# for ax in axes:
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
fig.savefig(
    paths.figures / "20230707_RAqr_Halpha.pdf",
    dpi=300,
)

## zoom ins

fig, axes = pro.subplots(ncols=2, space=0, width="3.5in")

vmax=max(np.nanmax(halpha_frame), np.nanmax(hacont_frame))
im = axes[0].imshow(halpha_frame, extent=ext, norm=simple_norm(halpha_frame, stretch="linear"), vmin=0, vmax=vmax)
axes[0].text(0.05, 0.05, r"Halpha", transform="axes", c="white", fontsize=9)
im = axes[1].imshow(hacont_frame, extent=ext, norm=simple_norm(hacont_frame, stretch="linear"), vmin=0, vmax=vmax)
axes[1].text(0.05, 0.05, r"Ha-Cont", transform="axes", c="white", fontsize=9)



# fig.colorbar(im, loc="r", label=r"mJy / sq. arcsec")

## sup title
axes.format(
    xlim=(0.12, -0.12),
    ylim=(-0.12, 0.12),
    # suptitle=f"2023/07/07 VAMPIRES HD 169142"
)
axes.format(
    xlabel='$\Delta$RA (")',
    ylabel='$\Delta$DEC (")',
)
# for ax in axes:
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
# axes[1].format(yticks=[], ylabel="")
fig.savefig(
    paths.figures / "20230707_RAqr_mosaic.pdf",
    dpi=300,
)