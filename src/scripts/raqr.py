import paths
import proplot as pro
import numpy as np
from astropy.io import fits
from matplotlib import patches
from astropy.visualization import simple_norm

pro.rc["legend.fontsize"] = 7
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"
pro.rc["axes.grid"] = False
# pro.rc["cycle"] = "ggplot"


fig, axes = pro.subplots(
    width="3.5in",
)

halpha_frame, header = fits.getdata(paths.data / "20230707_RAqr_Halpha.fits", header=True)
hacont_frame = fits.getdata(paths.data / "20230707_RAqr_Ha-cont.fits")
halpha_frame = np.flipud(halpha_frame)
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
    # rect = patches.Rectangle([-0.5, -0.5], bar_width_arc, 1e-2, color="white")
    # ax.add_patch(rect)
    # ax.text(-0.5 + bar_width_arc/2, -0.45, f"{bar_width_au:.0f} au", c="white", ha="center", fontsize=7)



# fig.colorbar(im, loc="r", label=r"mJy / sq. arcsec")

## sup title
axes.format(
    xlim=(0.6, -0.6),
    ylim=(-1.1, 0.8),
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
    paths.figures / "20230707_RAqr_Halpha.pdf",
    dpi=300,
)
