import paths
import proplot as pro
import numpy as np

pro.rc["legend.fontsize"] = 7
pro.rc["cycle"] = "ggplot"
pro.rc["font.size"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["axes.grid"] = False


def get_emccd_noise(photons, texp, emgain=300):
    dark_e = 1.5e-4 * texp
    if emgain > 1:
        enf = np.sqrt(2)  # theoretical
        rn_e = 89 / emgain
        fullwell_e = 800_000 / emgain
    else:
        enf = 1
        rn_e = 9.6
        fullwell_e = 180_000
    cic_flux = 1e-3  # e- / pix / frame
    # open filter QE
    ave_qe = 0.85
    fullwell = min(fullwell_e, (2**16 - 150) * 4.5)
    signal = photons * ave_qe * texp
    total_signal = signal + dark_e + cic_flux
    total_signal[total_signal > fullwell] = np.nan
    noise_e = np.sqrt(total_signal * enf**2 + rn_e**2)
    return noise_e


def get_emccd_snr(photons, texp, emgain=300):
    ave_qe = 0.85
    signal = photons * texp
    noise_e = get_emccd_noise(photons, texp, emgain)
    return signal / noise_e


def get_cmos_noise(photons, texp, mode: str):
    dark_e = 3.6e-3 * texp
    if mode == "fast":
        rn_e = 0.4
        fullwell = 2**16 * 0.103
    elif mode == "slow":
        rn_e = 0.235
        fullwell = 2**16 * 0.105
    # open filter QE
    ave_qe = 0.678
    signal = photons * ave_qe * texp
    total_signal = signal + dark_e
    total_signal[total_signal > fullwell] = np.nan
    return np.sqrt(total_signal + rn_e**2)


def get_cmos_snr(photons, texp, mode: str = "slow"):
    ave_qe = 0.678
    signal = photons * texp
    noise_e = get_cmos_noise(photons, texp, mode)
    return signal / noise_e

photons = np.geomspace(1e-3, 10, 10000)
texp = np.geomspace(1, 1800, 10000)
photons_grid, texp_grid = np.meshgrid(photons, texp)


snr_emccd = get_emccd_snr(photons_grid, texp_grid, emgain=300)
snr_cmos = get_cmos_snr(photons_grid, texp_grid, mode="slow")

ratio = (snr_cmos / snr_emccd)

fig, axes = pro.subplots(width="3.5in", refheight="1.5in", aspect="equal")

ext = (photons.min(), photons.max(), texp.min(), texp.max())
norm = pro.DivergingNorm(vcenter=1)
# norm = None
cm = axes[0].imshow(ratio, cmap="curl", norm=norm, extent=ext)
axes[0].colorbar(cm, label="relative S/N (qCMOS / EMCCD)")


# cm = axes[1].imshow(snr_cmos, cmap="magma", norm=norm, extent=ext)
# axes[1].colorbar(cm, label="relative S/N (CMOS / EMCCD)")

# for i, texp in enumerate((0.1, 100)):
#     benchmark = photons / np.sqrt(photons)
#     axes[i].axhline(1, color="0.2", label="Ideal camera")
#     axes[i].plot(
#         photons,
#         get_cmos_snr(photons / texp, texp, mode="slow") / benchmark,
#         c="C0",
#         label="qCMOS (SLOW)",
#         zorder=10,
#     )
#     axes[i].plot(
#         photons,
#         get_cmos_snr(photons / texp, texp, mode="fast") / benchmark,
#         c="C0",
#         ls="--",
#         label="qCMOS (FAST)",
#         zorder=9,
#     )
#     axes[i].plot(
#         photons,
#         get_emccd_snr(photons / texp, texp) / benchmark,
#         c="C3",
#         label="EMCCD (g=300)",
#     )
#     axes[i].plot(
#         photons,
#         get_emccd_snr(photons / texp, texp, emgain=0) / benchmark,
#         c="C3",
#         ls="--",
#         label="EMCCD (g=0)",
#     )
#     axes[i].format(title=f"DIT={texp} s")
#     axes[i].legend(ncols=1, frame=False)
axes[0].format(title="SLOW vs. g=300", titleloc="ll", titlecolor="k", titleborder=False)
axes.format(
    xlabel="photons / pixel / frame",
    ylabel="DIT (s)",
    yformatter="log",
    yscale="log",
    xformatter="log",
    xscale="log",
    aspect="auto"
)

# save output
fig.savefig(paths.figures / "detector_snr_surface.pdf", dpi=300)
