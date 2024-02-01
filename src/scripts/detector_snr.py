import paths
import proplot as pro
import numpy as np

pro.rc["legend.fontsize"] = 7
pro.rc["cycle"] = "ggplot"
pro.rc["font.size"] = 8


def get_emccd_noise(photons, texp, emgain=300):
    dark_e = 1.5e-4 * texp
    if emgain > 1:
        enf = np.sqrt(2)  # theoretical
        rn_e = 89 / emgain
        fullwell_e = 800_000
    else:
        enf = 1
        rn_e = 9.6
        fullwell_e = 180_000
    cic_flux = 1e-3  # e- / pix / frame
    # open filter QE
    ave_qe = 0.85
    fullwell = min(fullwell_e, (2**16 - 150) * 4.5)
    signal = photons * ave_qe * texp
    signal[signal > fullwell] = np.nan
    noise_e = np.sqrt(signal * enf**2 + rn_e**2 + dark_e * enf**2 + cic_flux * enf**2)
    return noise_e

def get_emccd_snr(photons, texp, emgain=300):
    ave_qe = 0.85
    signal = photons * ave_qe * texp
    noise_e = get_emccd_noise(photons, texp, emgain)
    return signal / noise_e

def get_emccd_dr(emgain=300):
    if emgain > 1:
        fullwell = min(800_000, (2**15 - 150) * 4.5) / emgain
        rn_e = 89 / emgain
    else:
        fullwell = min(180_000, (2**15 - 150) * 4.5)
        rn_e = 9.6
    cic_flux = 1e-3  # e- / pix / frame
    equiv_noise = np.sqrt(rn_e**2 + cic_flux)
    return 20 * np.log10(fullwell / equiv_noise)


def get_cmos_noise(photons, texp, mode:str):
    dark_e = 3.6e-3 * texp
    if mode == "fast":
        rn_e = 0.4
        fullwell = 2**16 * 0.103
    elif mode == "slow":
        rn_e = 0.235
        fullwell = 2**16 * 0.105
    # open filter QE
    ave_qe = 0.678
    electrons = photons * ave_qe * texp
    electrons[electrons > fullwell] = np.nan
    return np.sqrt(electrons + rn_e**2 + dark_e)

def get_cmos_snr(photons, texp, mode: str = "slow"):
    ave_qe = 0.678
    electrons = photons * ave_qe * texp
    noise_e = get_cmos_noise(photons, texp, mode)
    return electrons / noise_e


def get_cmos_dr(mode: str = "slow"):
    if mode == "fast":
        rn_e = 0.4
        fullwell = 2**16 * 0.103
    elif mode == "slow":
        rn_e = 0.235
        fullwell = 2**16 * 0.105
    return 20 * np.log10(fullwell / rn_e)


# set min photons where qCMOS SLOW S/N == 1 (RN limited)
min_photons = 0.5 / 0.678 * (1 + np.hypot(1, 2 * 0.235)) / 100
photons = np.geomspace(1e-3, 1e6, 10_000)

fig, axes = pro.subplots(nrows=2, width="3.5in", refheight="1.5in")

for i, texp in enumerate((0.1, 100)):
    benchmark = photons / np.sqrt(photons)
    axes[i].axhline(1, ls=":", color="0.3", label="Ideal camera")
    axes[i].plot(
        photons,
        get_cmos_snr(photons / texp, texp, mode="slow") / benchmark,
        c="C0",
        label="qCMOS (SLOW)",
        zorder=10,
    )
    axes[i].plot(
        photons,
        get_cmos_snr(photons / texp, texp, mode="fast") / benchmark,
        c="C0",
        ls="--",
        label="qCMOS (FAST)",
        zorder=9,
    )
    axes[i].plot(
        photons,
        get_emccd_snr(photons / texp, texp) / benchmark,
        c="C3",
        label="EMCCD (g=300)",
    )
    axes[i].plot(
        photons,
        get_emccd_snr(photons / texp, texp, emgain=0) / benchmark,
        c="C3",
        ls="--",
        label="EMCCD (g=0)",
    )
    axes[i].format(title=f"DIT={texp} s")
    axes[i].legend(ncols=1)

axes.format(
    xlabel="photons / pixel / frame",
    ylabel="relative S/N",
    xscale="log",
    xformatter="log",
)

# save output
fig.savefig(paths.figures / "detector_snr_relative.pdf", dpi=300)
