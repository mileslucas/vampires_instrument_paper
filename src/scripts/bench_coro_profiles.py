import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from photutils.profiles import RadialProfile
from astropy.convolution import kernels, convolve_fft
from matplotlib import ticker

pro.rc["legend.fontsize"] = 6
pro.rc["legend.frameon"] = True
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8

mask_names = {
    "Fieldstop": "PSF",
    "CLC-2": "CLC-2",
    "CLC-3": "CLC-3",
    "CLC-5": "CLC-5",
    "CLC-7": "CLC-7",
    "DGVVC": "DGVVC",
}

transmission = {
    ("Open", "Open"): 1.0,
    ("Open", "OD 0.3"): 0.62,
    ("Open", "OD 1.0"): 0.165,
    ("Open", "OD 2.0"): 0.034,
    ("Open", "OD 3.0"): 0.0078,
    ("OD 4.0 (vis)", "Open"): 0.001331,
    ("OD 4.0 (vis)", "OD 0.3"): 8.18e-4,
    ("OD 4.0 (vis)", "OD 1.0"): 2.2e-4,
    ("OD 4.0 (vis)", "OD 2.0"): 4.9e-5,
    ("OD 4.0 (vis)", "OD 3.0"): 1.2e-5,
}

stop_thru = {
    "Open": 1,
    "PupilRef": 0.982,
    "LyotOpt": 0.894,
    "LyotStop": 0.661,
}
kernel = kernels.Tophat2DKernel(radius=1)
data_dict = {}
for path in (paths.data / "coro_images").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    if hdr["FILTER01"] != "Open" or hdr["U_MASK"] == "RAP":
        continue
    key = hdr["U_FLDSTP"], hdr["U_MASK"]
    through = (
        transmission[(hdr["X_SRCFFT"], hdr["X_SRCFOP"])] * stop_thru[hdr["U_MASK"]]
    )
    # print(hdr["U_SRCFLX"])
    data_dict[key] = convolve_fft(np.squeeze(data) / through, kernel)

for path in (paths.data / "psf_images").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    if hdr["FILTER01"] != "Open" or hdr["OBS-MOD"] != "IMAG":
        continue
    key = hdr["U_FLDSTP"], hdr["U_MASK"]
    through = transmission[(hdr["X_SRCFFT"], hdr["X_SRCFOP"])]
    data_dict[key] = convolve_fft(np.squeeze(data) / through, kernel)


plate_scale = 6.03  # mas / px

fig, axes = pro.subplots(nrows=4, space=0, width="3.5in", height="4.5in")

rads = np.arange(0, 1e3 / plate_scale)

cycle = pro.Colormap("boreal")(np.linspace(0.3, 0.7, len(mask_names) - 1))
cycle = ["0.2", *cycle]
cy, cx = np.array(data_dict["Fieldstop", "Open"].shape[-2:]) / 2 - 0.5
radprof_psf = RadialProfile(data_dict["Fieldstop", "Open"], (cx, cy), rads)
norm_val = np.nanmax(radprof_psf.profile)
iwas = {"CLC-2": 37, "CLC-3": 59, "CLC-5": 105, "CLC-7": 150, "DGVVC": 61}
stop_names = {
    "Open": "No stop",
    "PupilRef": "LyotStop-S",
    "LyotOpt": "LyotStop-M",
    "LyotStop": "LyotStop-L",
}

for fieldstop, color in zip(mask_names, cycle):
    # 1 radial profile
    for ax, stop in zip(axes, stop_names):
        if fieldstop == "Fieldstop":
            key = fieldstop, "Open"
        else:
            key = fieldstop, stop
        cy, cx = np.array(data_dict[key].shape[-2:]) / 2 - 0.5
        radprof = RadialProfile(data_dict[key], (cx, cy), rads)
        # plot data
        ax.plot(
            radprof.radii[:-1] * plate_scale / 1e3,
            radprof.profile / norm_val,
            label=mask_names[fieldstop],
            c=color,
            lw=1,
            zorder=900 if fieldstop == "Fieldstop" else 800,
        )
        if fieldstop in iwas:
            ax.axvline(iwas[fieldstop] / 1e3, c=color, ls=":", lw=1, alpha=0.9)
        ax.text(
            0.98,
            0.33,
            stop_names[stop],
            transform="axes",
            va="bottom",
            ha="right",
            fontsize=7,
            c="0.3",
        )
ave_wave = 680e-9
ave_lamd = np.rad2deg(ave_wave / 7.92) * 3.6e3
axes[0].dualx(lambda x: x / ave_lamd, label=r"separation ($\lambda/D$)")


axes[0].legend(ncols=2, order="F")
# axes[1].legend(ncols=2)
axes.format(
    grid=True,
    xlabel='separation (")',
    
)
axes.format(
    ylabel="normalized profile",
    yscale="log",
    yformatter="log",
    xlim=(0, 0.4),
    xlocator=ticker.MaxNLocator(5, prune="both"),
)
# save output
fig.savefig(paths.figures / "bench_coro_profiles.pdf", dpi=300)
