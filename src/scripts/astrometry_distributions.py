import paths
import proplot as pro
import numpy as np
from uncertainties import ufloat, unumpy as unp
from astropy.stats import circstats

pro.rc["legend.fontsize"] = 8
pro.rc["font.size"] = 9
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"
pro.rc["lines.markersize"] = 3

# hd1160_sept = {"sep": ufloat(0.79305, 0.0025), "pa": ufloat(244.486, 0.170)}
hd1160_sept = {"sep": ufloat(794.346e-3, 8.171e-3), "pa": ufloat(244.304, 0.388)}
hd1160_meas = {"sep": ufloat(135.181, 0.052), "pa": ufloat(245.073, 0.017)}
derot_offset = -40.9 + 180 - 39
hd1160_meas["pa"] -= derot_offset
hd1160_res = {
    "sep": hd1160_sept["sep"] * 1e3 / hd1160_meas["sep"],
    "pa": hd1160_sept["pa"] - hd1160_meas["pa"],
}

cam1_results = {
    "HD 139341": {"sep": ufloat(5.916, 0.016), "pa": ufloat(102.8, 0.23)},
    "HD 137909": {"sep": ufloat(5.86, 0.04), "pa": ufloat(102.41, 0.24)},
    "HD 1160": hd1160_res,
    "HIP 3373": {"sep": ufloat(5.89, 0.28), "pa": ufloat(99.0, 3.1)},
    "Albireo": {"sep": ufloat(6.03, 0.15), "pa": ufloat(103, 6)},
    "21 Oph": {"sep": ufloat(6.03, 0.29), "pa": ufloat(100.4, 3.1)},
}
cam2_results = {
    "21 Oph": {"sep": ufloat(6.03, 0.29), "pa": ufloat(99.5, 3.1)},
    "Albireo": {"sep": ufloat(6.04, 0.15), "pa": ufloat(102, 6)},
    # "HD 1160": {"sep": ufloat(np.nan, 0.28), "pa": ufloat(np.nan, 3.1)},
    "HIP 3373": {"sep": ufloat(5.86, 0.28), "pa": ufloat(98.2, 3.1)},
    "HD 137909": {"sep": ufloat(5.80, 0.04), "pa": ufloat(102.12, 0.24)},
    "HD 139341": {"sep": ufloat(5.908, 0.016), "pa": ufloat(102.73, 0.23)},
}


def _weighted_mean(values):
    means = unp.nominal_values(values)
    stds = unp.std_devs(values)
    weights = 1 / stds**2
    mu = np.sum(means * weights) / weights.sum()
    sigma = np.sqrt(1 / np.sum(weights))
    return ufloat(mu, sigma)


def _weighted_circular_mean(values):
    ang_rads = values * np.pi / 180
    weights = 1 / unp.std_devs(ang_rads) ** 2
    mean_rad = circstats.circmean(unp.nominal_values(ang_rads), weights=weights)
    sig_rad = np.sqrt(1 / np.sum(weights))
    return ufloat(mean_rad, sig_rad) * 180 / np.pi


def weighted_mean(results: dict):
    seps = np.array([res["sep"] for res in results.values()])
    pas = np.array([res["pa"] for res in results.values()])
    output = {"sep": _weighted_mean(seps), "pa": _weighted_circular_mean(pas)}
    output["pad"] = output["pa"] + 219 - 360
    return output


cam1_mean = weighted_mean(cam1_results)
cam2_mean = weighted_mean(cam2_results)
print(f"VCAM1: {cam1_mean}")
print(f"VCAM2: {cam2_mean}")
print(
    (cam1_mean["pa"] - cam1_results["HD 1160"]["pa"]).nominal_value
    / cam1_results["HD 1160"]["pa"].std_dev
)

fig, axes = pro.subplots(
    nrows=2, ncols=2, width="3.5in", height="2.3in", spanx=False, hspace=0, wspace=0.5
)


ys = dict(zip(cam1_results.keys(), range(len(cam1_results))))
for key, y in ys.items():
    axes[0, 0].errorbar(
        cam1_results[key]["sep"].nominal_value,
        y=y,
        xerr=cam1_results[key]["sep"].std_dev,
        c="C0",
        marker="o",
        zorder=100,
    )
    axes[0, 1].errorbar(
        cam1_results[key]["pa"].nominal_value,
        y=y,
        xerr=cam1_results[key]["pa"].std_dev,
        c="C1",
        marker="o",
        zorder=100,
    )
    if key in cam2_results:
        axes[1, 0].errorbar(
            cam2_results[key]["sep"].nominal_value,
            y=y,
            xerr=cam2_results[key]["sep"].std_dev,
            c="C0",
            marker="^",
            zorder=100,
        )
        axes[1, 1].errorbar(
            cam2_results[key]["pa"].nominal_value,
            y=y,
            xerr=cam2_results[key]["pa"].std_dev,
            c="C1",
            marker="^",
            zorder=100,
        )

ymin = -0.75
ymax = len(cam1_results) - 1 + 0.75
axes[0, 0].axvline(cam1_mean["sep"].nominal_value, c="0.3")
axes[0, 0].fill_between(
    [
        cam1_mean["sep"].nominal_value - cam1_mean["sep"].std_dev,
        cam1_mean["sep"].nominal_value + cam1_mean["sep"].std_dev,
    ],
    ymin,
    ymax,
    c="0.3",
    alpha=0.2,
    zorder=0,
)
axes[0, 1].axvline(cam1_mean["pa"].nominal_value, c="0.3")
axes[0, 1].fill_between(
    [
        cam1_mean["pa"].nominal_value - cam1_mean["pa"].std_dev,
        cam1_mean["pa"].nominal_value + cam1_mean["pa"].std_dev,
    ],
    ymin,
    ymax,
    c="0.3",
    alpha=0.2,
    zorder=0,
)
axes[1, 0].axvline(cam2_mean["sep"].nominal_value, c="0.3")
axes[1, 0].fill_between(
    [
        cam2_mean["sep"].nominal_value - cam2_mean["sep"].std_dev,
        cam2_mean["sep"].nominal_value + cam2_mean["sep"].std_dev,
    ],
    ymin,
    ymax,
    c="0.3",
    alpha=0.2,
    zorder=0,
)
axes[1, 1].axvline(cam2_mean["pa"].nominal_value, c="0.3")
axes[1, 1].fill_between(
    [
        cam2_mean["pa"].nominal_value - cam2_mean["pa"].std_dev,
        cam2_mean["pa"].nominal_value + cam2_mean["pa"].std_dev,
    ],
    ymin,
    ymax,
    c="0.3",
    alpha=0.2,
    zorder=0,
)

axes[1, 0].format(xlabel="plate scale (mas/px)")
axes[1, 1].format(xlabel="PA offset (°)")
axes[:, 0].format(yformatter=list(ys.keys()), ylocator=list(ys.values()))

axes[:, 1].format(ytickloc="none")

axes.format(
    ylim=(ymin, ymax),
    rightlabels=("VCAM1", "VCAM2"),
    rightlabelpad=2,
    rightlabels_kw=dict(rotation=-90),
)

fig.savefig(paths.figures / "astrometry_results.pdf", dpi=300)
