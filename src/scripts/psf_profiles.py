import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from photutils.profiles import RadialProfile, CurveOfGrowth
from astropy.convolution import kernels, convolve_fft
from matplotlib import ticker

pro.rc["legend.fontsize"] = 6
pro.rc["legend.frameon"] = True
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8

filters = (
    "625-50",
    "675-50",
    "725-50",
    "750-50",
    "775-50",
    "Open",
)
nb_filters = (
    "Ha-Cont",
    "Halpha",
    "SII",
    "SII-Cont"
)
mbi_filters = (
    "F610",
    "F670",
    "F720",
    "F760"
)

# filts = (
#     "625-50",
#     "675-50",
#     "725-50",
#     "750-50",
#     "775-50",
#     "Open",
# )


data_dict = {}
for path in (paths.data / "psf_images").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    if hdr["FILTER02"] != "Open":
        data_dict[hdr["FILTER02"]] = np.squeeze(data)
    elif "MBI" in hdr["OBS-MOD"]:
        for frame, f in zip(data, mbi_filters):
            data_dict[f] = frame
    else:
        data_dict[hdr["FILTER01"]] = np.squeeze(data)

kern = kernels.Tophat2DKernel(radius=1)
for k in data_dict:
    data_dict[k] = convolve_fft(data_dict[k], kern)

plate_scale = 6.03 # mas / px

fig, axes = pro.subplots(
    nrows=2,
    width="3.5in",
    height="3.25in",
    hspace=0.4,
    sharey=0,
    hratios=(0.7, 0.3)
)


rads = np.arange(0, 1.05e3 / plate_scale)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, len(filters) - 1))
cycle = [*cycle, "0.1"]

for filt, color in zip(filters, cycle):
    # 1 radial profile
    cy, cx = np.array(data_dict[filt].shape[-2:]) / 2 - 0.5
    radprof = RadialProfile(data_dict[filt], (cx, cy), rads)
    radprof.normalize()
    # plot data
    axes[0].plot(
        radprof.radius * plate_scale / 1e3,
        radprof.profile,
        label=f"{filt}",
        c=color,
        lw=1,
    )

for filt, color in zip(filters, cycle):
    # 1 radial profile
    cy, cx = np.array(data_dict[filt].shape[-2:]) / 2 - 0.5
    cog = CurveOfGrowth(data_dict[filt], (cx, cy), rads[1:])
    cog.normalize()
    # plot data
    axes[1].plot(
        [0, *(cog.radius * plate_scale / 1e3).tolist()],
        [0, *cog.profile.tolist()],
        label=f"{filt}",
        c=color,
        lw=1,
    )


ave_wave = 680e-9
ave_lamd = np.rad2deg(ave_wave / 7.92)*3.6e3
axes[0].dualx(lambda x: x / ave_lamd, label=r"separation ($\lambda_\mathrm{Open}/D$)")

for ax in axes:
    ax.axvline(17/2 * ave_lamd, c="0.5", ls=":", lw=0.7, zorder=0)
    ax.axvline(46.6/2 * ave_lamd, c="0.5", ls="-", lw=0.7, zorder=0)
    ax.axvline(68/2 * ave_lamd, c="0.5", ls="--", lw=0.7, zorder=0)

axes[0].text(17/2 * ave_lamd + 6e-3, 1, "AO188 control radius", c="0.5", rotation=-90, fontsize=6, va="top")
axes[0].text(46.6/2 * ave_lamd + 6e-3, 1, "SCExAO control radius", c="0.5", rotation=-90, fontsize=6, va="top")
axes[0].text(68/2 * ave_lamd + 6e-3, 1, "AO3K control radius", c="0.5", rotation=-90, fontsize=6, va="top")

axes[0].legend(ncols=1)
# axes[1].legend(ncols=2)
axes.format(
    grid=True,
    xlabel="separation (\")",
)
axes[0].format(
    ylabel="normalized profile",
    yscale="log",
    yformatter="log",
    xtickloc="top",
)
axes[1].format(
    ylabel="encircled energy",
    ylim=(0, None)
)
# save output
fig.savefig(paths.figures / "bench_psf_profiles.pdf", dpi=300)
