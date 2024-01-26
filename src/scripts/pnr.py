import paths
import proplot as pro
import numpy as np
from scipy.special import erf

pro.rc["legend.fontsize"] = 7
pro.rc["cycle"] = "ggplot"

ave_qe = 0.678

def theoretical_noise(flux, mode="slow"):
    if mode == "fast":
        rn_e = 0.4
    elif mode == "slow":
        rn_e = 0.235
    noise_e = np.sqrt(flux * ave_qe + rn_e**2)
    return noise_e

def phi(z):
    return 0.5 * (1 + erf(z/np.sqrt(2)))

def vince_rn(k, sigma_rn):
    left_side = phi((k + 0.5) / sigma_rn)
    right_side =  phi((k - 0.5) / sigma_rn)
    summand = k**2 * (left_side - right_side)
    return np.sqrt(np.sum(summand))

def theoretical_pnr_noise(flux, mode="slow"):
    k = np.arange(-100, 101)
    if mode == "fast":
        rn_e =  vince_rn(k, 0.4)
    elif mode == "slow":
        rn_e =  vince_rn(k, 0.235)
    noise_e = np.sqrt(flux * ave_qe + rn_e**2)
    return noise_e

def montecarlo_noise(flux, mode="slow", N=100000):
    if mode == "fast":
        rn_e = 0.4
    elif mode == "slow":
        rn_e = 0.235

    samples = np.random.normal(np.random.poisson(flux * ave_qe, N), scale=rn_e)
    return np.std(samples)


def montecarlo_pnr_noise(flux, mode="slow", N=100000):
    if mode == "fast":
        rn_e = 0.4
    elif mode == "slow":
        rn_e = 0.235
    samples = np.random.normal(np.random.poisson(flux * ave_qe, N), scale=rn_e)
    return np.std(np.round(samples))


flux = np.geomspace(1e-4, 10, 100)

photon_noise = np.sqrt(flux * ave_qe)

fig, axes = pro.subplots(width="3.5in", height="2.5in")

theoretical_improvement_slow = theoretical_noise(flux) / theoretical_pnr_noise(flux)
theoretical_improvement_fast = theoretical_noise(flux, mode="fast") / theoretical_pnr_noise(flux, mode="fast")
mc_improvement_slow = np.vectorize(montecarlo_noise)(flux) / np.vectorize(montecarlo_pnr_noise)(flux)
mc_improvement_fast = np.vectorize(lambda f: montecarlo_noise(f, mode="fast"))(flux) / np.vectorize(lambda f: montecarlo_pnr_noise(f, mode="fast"))(flux)

# axes[0].plot(flux, flux / photon_noise, c="k", label="ideal")

axes[0].plot(flux, theoretical_improvement_slow , c="C0", zorder=999, label="Theoretical (Slow)")
axes[0].plot(flux, theoretical_improvement_fast , c="C0", zorder=999, ls="--", label="Theoretical (Fast)")
axes[0].plot(flux, mc_improvement_slow , c="C3", zorder=900, label="MC (Slow)")
axes[0].plot(flux, mc_improvement_fast , c="C3", zorder=900, ls="--", label="MC (Fast)")
axes[0].axhline(0, c="k", label="ideal")
axes[0].legend(ncols=1)

axes.format(
    xlabel="photons/pixel/frame",
    ylabel="relative S/N improvement (%)",
    xscale="log",
    xformatter="log"
)

# save output
fig.savefig(paths.figures / "pnr_improvement.pdf", dpi=300)
