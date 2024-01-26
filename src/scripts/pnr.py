import paths
import proplot as pro
import numpy as np
from scipy.special import erf
from scipy import stats
from scipy.optimize import minimize
from astropy.io import fits

pro.rc["legend.fontsize"] = 7
pro.rc["cycle"] = "ggplot"
pro.rc["image.origin"] = "lower"

vcam_rn = {
    "fast": 0.4,  # e- average of both cams
    "slow": 0.235,
}


data = fits.getdata(paths.data / "20230620_dark_vcam1_504s.fits")
data = data[:, :100, :100]

def theoretical_noise(flux, sigma_rn):
    return np.sqrt(flux + sigma_rn**2)


def phi(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def pnr_equiv_rn(sigma_rn, max_k=101):
    _sum = 0
    for k in np.arange(-max_k, max_k + 1):
        left_side = phi((k + 0.5) / sigma_rn)
        right_side = phi((k - 0.5) / sigma_rn)
        _sum += k**2 * (left_side - right_side)
    return np.sqrt(_sum)


def montecarlo_samples(
    flux, sigma_rn: float, N: int = 10000, rng=np.random.default_rng()
):
    return np.array([rng.normal(rng.poisson(f, size=N), scale=sigma_rn) for f in flux])

flux = np.geomspace(1e-3, 10, 500)  # photons/pix/frame

sigma_rn = np.linspace(0, 1, 500)

flux_grid, sigma_grid = np.meshgrid(flux, sigma_rn)


theoretical_improvement = theoretical_noise(flux_grid, sigma_grid) / theoretical_noise(
    flux_grid, pnr_equiv_rn(sigma_grid)
)

# fig, axes = pro.subplots(refwidth="3.5in", refheight="3.5in")


# im = axes[0].imshow(
#     theoretical_improvement - 1,
#     aspect="auto",
#     extent=(flux_grid.min(), flux_grid.max(), sigma_grid.min(), sigma_grid.max()),
#     cmap="curl",
#     norm="symlog",
#     norm_kw=dict(linthresh=0.01)
# )
# axes[0].colorbar(im, label="relative S/N improvement", formatter="log")

# axes[0].axhline(vcam_rn["slow"], c="k", ls="--", label="Slow")
# axes[0].axhline(vcam_rn["fast"], c="k", ls=":", label="Fast")
# mask = (
#     np.isclose(theoretical_improvement, 1, atol=1e-2)
#     & (sigma_grid > 0.1)
#     & (flux_grid < 1e-1)
# )
# turnover_point = np.median(sigma_grid[mask])
# axes[0].axhline(turnover_point, c="k", label=f"{turnover_point:.01f} e-")
# axes[0].legend(ncols=1)

# axes[0].format(
#     ylabel=r"$\sigma_{RN}$ (e-)",
#     xlabel="e- / pixel / frame",
#     xformatter="log",
#     xscale="log",
# )

# fig.savefig(paths.figures / "pnr.pdf", dpi=300)


fig, axes = pro.subplots(nrows=2, width="3.5in", height="4.5in", share=0)


axes[0].hist(data.ravel(), bins=np.arange(193, 261))
axes[0].format(
    xlabel="signal [adu]",
    ylabel="count",
)
axes[0].dualx(lambda adu: (adu - 200) * 0.105, label="signal [e-]")


theoretical_curves = {}
mc_curves = {}
for mode, sigma in vcam_rn.items():
    theoretical_improvement = theoretical_noise(flux, sigma) / theoretical_noise(
        flux, pnr_equiv_rn(sigma)
    )
    samples = montecarlo_samples(flux, sigma)
    mc_improvement = np.std(samples, axis=1) / np.std(np.round(samples), axis=1)
    theoretical_curves[mode] = theoretical_improvement
    mc_curves[mode] = mc_improvement

axes[1].plot(
    flux,
    (theoretical_curves["slow"] - 1) * 100,
    c="C0",
    zorder=999,
    lw=1,
    label="Theoretical (slow)",
)
axes[1].plot(
    flux,
    (theoretical_curves["fast"] - 1) * 100,
    c="C3",
    zorder=999,
    lw=1,
    label="Theoretical (fast)",
)
axes[1].scatter(
    flux,
    (mc_curves["slow"] - 1) * 100,
    c="C0",
    alpha=0.5,
    zorder=900,
    ms=5,
    mew=0,
    label="Monte Carlo (slow)",
)
axes[1].scatter(
    flux,
    (mc_curves["fast"] - 1) * 100,
    c="C3",
    alpha=0.5,
    zorder=900,
    ms=5,
    mew=0,
    label="Monte Carlo (fast)",
)
axes[1].legend(ncols=1)
axes[1].axhline(0, c="0.3")

axes[1].format(
    xlabel="e- / pixel / frame",
    ylabel="relative S/N improvement (%)",
    xscale="log",
    xformatter="log",
)


# save output
fig.savefig(paths.figures / "pnr.pdf", dpi=300)
