import paths
import proplot as pro
import numpy as np

pro.rc["legend.fontsize"] = 7
pro.rc["cycle"] = "ggplot"


def get_ccd_snr(photons, texp):
    dark_e = 1.5e-4 * texp
    rn_e = 9.6
    # open filter QE
    ave_qe = 0.85
    fullwell = min(180_000, (2**16 - 150) * 4.5)

    electrons = photons * ave_qe * texp
    electrons[electrons > fullwell] = np.nan
    noise_e = np.sqrt(electrons + rn_e**2 + dark_e)
    return electrons / noise_e

def get_ccd_dr():
    rn_e = 9.6
    fullwell =  min(180_000, (2**16 - 150) * 4.5)
    return 20 * np.log10(fullwell / rn_e)

def get_emccd_snr(photons, texp, emgain=300):
    dark_e = 1.5e-4 * texp
    enf = np.sqrt(2) # theoretical
    rn_e = 89 / emgain
    # open filter QE
    ave_qe = 0.85
    fullwell = min(800_000, (2**16 - 150) * 4.5)
    electrons = photons * ave_qe * texp
    electrons[electrons > fullwell] = np.nan
    noise_e = np.sqrt(electrons * enf**2 + rn_e**2 + dark_e * enf**2)
    return electrons / noise_e

def get_emccd_dr(emgain=300):
    rn_e = 89 / emgain
    fullwell =  min(800_000, (2**15 - 150) * 4.5) / emgain
    return 20 * np.log10(fullwell / rn_e)

def get_cmos_snr(photons, texp, mode: str = "slow"):
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
    noise_e = np.sqrt(electrons + rn_e**2 + dark_e)
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
min_photons = 0.5 / 0.678 * (1 + np.hypot(1, 2 * 0.235))
photons = np.geomspace(min_photons, 1e6, 10000)

fig, axes = pro.subplots(nrows=2, width="3.5in", refheight="1.5in")

ncoadds = 100
for i, texp in enumerate((0.1, 100)):
    cmos_slow_snr = get_cmos_snr(photons / texp, texp, mode="slow")
    cmos_slow_dr = get_cmos_dr(mode="slow")
    cmos_fast_snr = get_cmos_snr(photons / texp, texp, mode="fast")
    cmos_fast_dr = get_cmos_dr(mode="fast")
    emccd_snr = get_emccd_snr(photons / texp, texp)
    emccd_dr = get_emccd_dr()
    ccd_snr = get_ccd_snr(photons / texp, texp)
    ccd_dr = get_ccd_dr()
    axes[i].plot(photons, cmos_slow_snr, c="C0", label=f"CMOS (SLOW) DR={cmos_slow_dr:.0f} dB", zorder=10)
    axes[i].plot(photons, cmos_fast_snr, c="C0", ls="--", label=f"CMOS (FAST) DR={cmos_fast_dr:.0f} dB", zorder=9)
    axes[i].plot(photons, emccd_snr, c="C3", label=f"EMCCD (g=300) DR={emccd_dr:.0f} dB")
    axes[i].plot(photons, ccd_snr, c="C3", ls="--", label=f"EMCCD (g=0) DR={ccd_dr:.0f} dB")
    axes[i].format(
        title=f"DIT={texp} s",
    )
    axes[i].legend(ncols=1)

axes.format(
    xlabel="photons/pixel",
    ylabel="S/N",
    xscale="log",
    yscale="log",
    xformatter="log"
)

# save output
fig.savefig(paths.figures / "detector_snr.pdf", dpi=300)


fig, axes = pro.subplots(nrows=2, width="3.5in", refheight="1.5in")

for i, texp in enumerate((0.1, 100)):
    benchmark = get_cmos_snr(photons / texp, texp, mode="slow")
    axes[i].plot(photons, get_cmos_snr(photons / texp, texp, mode="slow") / benchmark, c="C0", label="CMOS (SLOW)", zorder=10)
    axes[i].plot(photons, get_cmos_snr(photons / texp, texp, mode="fast") / benchmark, c="C0", ls="--", label="CMOS (FAST)", zorder=9)
    axes[i].plot(photons, get_emccd_snr(photons / texp, texp) / benchmark, c="C3", label="EMCCD (g=300)")
    axes[i].plot(photons, get_ccd_snr(photons / texp, texp) / benchmark, c="C3", ls="--", label="EMCCD (g=0)")
    axes[i].format(
        title=f"DIT={texp} s"
    )
    axes[i].legend(ncols=1)

axes.format(
    xlabel="photons/pixel",
    ylabel="Relative S/N",
    xscale="log",
    xformatter="log",
)

# save output
fig.savefig(paths.figures / "detector_snr_relative.pdf", dpi=300)