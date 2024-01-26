import paths
import proplot as pro
import numpy as np
import pandas as pd
from scipy import interpolate

pro.rc["legend.fontsize"] = 7
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8

fig, axes = pro.subplots(
    width="3.5in",
    height="2.5in",
)

names = {
    "clc2": "CLC-2",
    "clc3": "CLC-3",
    "clc5": "CLC-5",
    "clc7": "CLC-7",
    "dgvvc": "DGVVC",
}
cycle = pro.Colormap("boreal")(np.linspace(0.3, 0.7, len(names)))
colors = {k: col for k, col in zip(names.keys(), cycle)}

plate_scale = 1.8  # arcsecond / mm
for key in names.keys():
    table = pd.read_csv(paths.data / "iwa_scans" / f"{key}.csv", index_col=0)
    table["x_arc"] = table["x"] * plate_scale
    table["y_arc"] = table["y"] * plate_scale
    table.sort_values(["x_arc", "y_arc"], inplace=True)

    # we took data at 3 y positions for every x position to try and alleviate any
    # small misalignment issues especially around the center of the dgVVC mask.
    # to reduce this, take the minimum flux along the y-axis
    reduced = table.groupby("x_arc").apply(lambda r: r["total"].min())

    radii = reduced.index
    flux = reduced.values

    # use the average flux for values where PSF is far from mask
    # as max flux to normalize flux from 0 to 1
    max_val = np.mean(flux[np.abs(radii) > 0.27])
    min_val = flux.min()
    flux_norm = (flux - min_val) / (max_val - min_val)

    # determine IWA where throughput is 0.5
    # we use a cubic spline interpolator

    # since interpolator has to have monotonic knots
    # create separately for positive  and negative radii
    mask_pos = radii >= 0
    mask_neg = radii <= 0
    itp_pos = interpolate.interp1d(flux_norm[mask_pos], radii[mask_pos], kind="cubic")
    itp_neg = interpolate.interp1d(
        flux_norm[mask_neg][::-1], np.abs(radii[mask_neg][::-1]), kind="cubic"
    )

    # get average IWA
    iwa_pos = itp_pos(0.5)
    iwa_neg = itp_neg(0.5)
    iwa = (iwa_pos + iwa_neg) / 2

    # find center of masked assuming it is symmetric
    # based on where the 15% crossings are
    idx_pos = np.where(flux_norm[mask_pos] >= 0.15)[0][0]
    idx_neg = np.where(flux_norm[mask_neg][::-1] >= 0.15)[0][0]
    rad_pos = radii[mask_pos][idx_pos]
    rad_neg = radii[mask_neg][::-1][idx_neg]
    # should be close to 0 since rad_neg is <0
    offset = (rad_pos + rad_neg) / 2
    radii_centered = radii - offset

    # finally, take average of negative and positive radii
    mask_pos = radii_centered >= 0
    mask_neg = radii_centered <= 0
    flux_iter = zip(flux_norm[mask_pos], flux_norm[mask_neg][::-1])
    rad_iter = zip(radii_centered[mask_pos], radii_centered[mask_neg][::-1])

    radii_ave = np.array([np.mean(np.abs(r)) for r in rad_iter])
    flux_ave = np.array([np.mean(flux) for flux in flux_iter])
    # plot data
    axes[0].plot(
        radii_ave * 1e3,  # convert to mas
        flux_ave,
        marker=".",
        label=f"{names[key]} ({iwa * 1e3:.0f} mas)",
        c=colors[key],
        lw=1,
        ms=5,
    )
    # vertical IWA line
    axes[0].axvline(iwa * 1e3, c=colors[key], ls=":")

# format and legends
axes.legend(ncols=1, title="Mask (IWA)")
axes.format(
    grid=True, xlabel="separation (mas)", ylabel="normalized throughput", xlim=(0, 255)
)

# save output
fig.savefig(paths.figures / "coronagraph_iwa.pdf", dpi=300)
