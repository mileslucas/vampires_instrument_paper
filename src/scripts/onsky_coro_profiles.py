import paths
import proplot as pro
from astropy.io import fits
import numpy as np
from photutils.profiles import RadialProfile

pro.rc["cycle"] = "ggplot"
pro.rc["image.origin"] = "lower"
pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8

with fits.open(paths.data / "HD102438_adi_cube.fits") as hdul:
    cube = hdul[0].data[159]
    header = hdul[0].header


with fits.open(paths.data / "20230711_HD1160" / "HD1160_adi_cube.fits") as hdul:
    cube2 = hdul[0].data[39]
    header2 = hdul[0].header


titles = ("F610", "F670", "F720", "F760")

fig, axes = pro.subplots(nrows=2, width="3.5in", height="3.5in", hspace=0.5)

plate_scale = 5.9e-3
side_length = cube.shape[-1] * plate_scale * 1e-3 / 2
ext = (side_length, -side_length, -side_length, side_length)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.9, len(titles)))

rads = np.arange(0, 0.8e3 / plate_scale)
for i in range(len(cube)):
    cy, cx = np.array(cube[i].shape[-2:]) / 2 - 0.5
    radprof = RadialProfile(cube[i], (cx, cy), rads)
    radprof.normalize()
    cy, cx = np.array(cube2[i].shape[-2:]) / 2 - 0.5
    radprof2 = RadialProfile(cube2[i], (cx, cy), rads)
    norm_val = radprof.profile[80]
    # plot data
    axes[0].plot(
        radprof.radii[:-1] * plate_scale / 1e3,
        radprof.profile,
        label=titles[i],
        c=cycle[i],
        lw=1,
    )
    axes[1].plot(
        radprof2.radii[:-1] * plate_scale / 1e3,
        radprof2.profile * norm_val / radprof2.profile[80],
        c=cycle[i],
        lw=1,
        label=titles[i],
    )
axes[0].axvline(59e-3, c="0.4", ls=":", lw=0.75, zorder=0)
axes[1].axvline(105e-3, c="0.4", ls=":", lw=0.75, zorder=0)
ave_lamd = header["RESELEM"] * 1e-3
[ax.axvline(ave_lamd * 46.6 / 2, c="0.4", lw=0.75, ls="--", zorder=0) for ax in axes]
[
    ax.text(
        ave_lamd * 46.6 / 2 + ave_lamd / 2,
        7e-3,
        "control radius",
        rotation=-90,
        fontsize=7,
        c="0.4",
        va="bottom",
        ha="left",
    )
    for ax in axes
]
axes[0].text(
    ave_lamd * 2.5 + ave_lamd,
    0.1,
    "IWA",
    rotation=-90,
    fontsize=7,
    c="0.4",
    va="center",
    ha="left",
)
axes[1].text(
    ave_lamd * 5 + ave_lamd,
    0.1,
    "IWA",
    rotation=-90,
    fontsize=7,
    c="0.4",
    va="center",
    ha="left",
)
axes[0].dualx(lambda x: x / ave_lamd, label=r"separation ($\lambda_{ave}/D$)")
axes[0].text(
    ave_lamd * 15.5 + ave_lamd,
    0.15,
    "calibration\nspeckles",
    va="top",
    ha="center",
    c="0.4",
    fontsize=7,
)
axes[0].text(
    ave_lamd * 48 + ave_lamd,
    0.05,
    "passive\nspeckles",
    va="bottom",
    ha="center",
    c="0.4",
    fontsize=7,
)
axes[0].legend(ncols=1)
axes.format(
    ylabel=r"normalized profile",
    yscale="log",
    yformatter="log",
    xlabel='separation (")',
    ylim=(6e-3, 1.5),
    rightlabels=("CLC-3", "CLC-5"),
    rightlabelpad=2,
    rightlabels_kw=dict(rotation=-90, fontweight="normal"),
)
axes[0].format(xtickloc="none")

fig.savefig(
    paths.figures / "onsky_coro_profiles.pdf",
    dpi=300,
)
