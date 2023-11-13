import paths
import proplot as pro
import numpy as np

pro.rc["legend.fontsize"] = 8
pro.rc["cycle"] = "538"


def get_ccd_snr(photons, texp):
    dark_e = 1.5e-4 * texp
    rn_e = 9.6
    # open filter QE
    ave_qe = 0.85
    fullwell = min(180_000, 2**16 * 4.5)

    electrons = photons * ave_qe * texp
    electrons[electrons > fullwell] = np.nan
    noise_e = np.sqrt(electrons + rn_e**2 + dark_e)
    return electrons / noise_e

def get_emccd_snr(photons, texp, emgain=300):
    dark_e = 1.5e-4 * texp
    enf = np.sqrt(2) # theoretical
    rn_e = 89 / emgain
    # open filter QE
    ave_qe = 0.85
    fullwell = min(800_000, 2**16 * 4.5)
    electrons = photons * ave_qe * texp
    electrons[electrons > fullwell] = np.nan
    noise_e = np.sqrt(electrons * enf**2 + rn_e**2 + dark_e * enf**2)
    return electrons / noise_e

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

photons = np.geomspace(1e-1, 1e6, 1000)

fig, axes = pro.subplots(nrows=2, width="3.5in", refheight="1.5in")


for i, texp in enumerate((0.1, 60)):
    axes[i].plot(photons, get_cmos_snr(photons / texp, texp, mode="slow"), c="C0", label="CMOS (SLOW)", zorder=10)
    axes[i].plot(photons, get_cmos_snr(photons / texp, texp, mode="fast"), c="C0", ls="--", label="CMOS (FAST)", zorder=9)
    axes[i].plot(photons, get_emccd_snr(photons / texp, texp), c="C1", label="CCD (EM=300)")
    axes[i].plot(photons, get_ccd_snr(photons / texp, texp), c="C1", ls="--", label="CCD (EM=0)")
    axes[i].format(
        title=f"DIT={texp} s",
        ylim=(1e-1, None)
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

for i, texp in enumerate((0.1, 60)):
    benchmark = get_emccd_snr(photons / texp, texp)
    axes[i].plot(photons, get_cmos_snr(photons / texp, texp, mode="slow") / benchmark, c="C0", label="CMOS (SLOW)", zorder=10)
    axes[i].plot(photons, get_cmos_snr(photons / texp, texp, mode="fast") / benchmark, c="C0", ls="--", label="CMOS (FAST)", zorder=9)
    axes[i].plot(photons, get_emccd_snr(photons / texp, texp) / benchmark, c="C1", label="CCD (EM=300)")
    axes[i].plot(photons, get_ccd_snr(photons / texp, texp) / benchmark, c="C1", ls="--", label="CCD (EM=0)")
    axes[i].format(
        title=f"DIT={texp} s"
    )
    axes[i].legend(ncols=1)

axes.format(
    xlabel="photons/pixel",
    ylabel="Relative S/N",
    xscale="log",
    xformatter="log"
)

# save output
fig.savefig(paths.figures / "detector_snr_relative.pdf", dpi=300)