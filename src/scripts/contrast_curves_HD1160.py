import paths
import proplot as pro
import numpy as np
import pandas as pd
from astropy.io import fits

pro.rc["legend.fontsize"] = 7
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"

all_filenames = (paths.data / "20230711_HD1160").glob("*.csv")
filenames_adi = sorted((paths.data / "20230711_HD1160").glob("*F*.csv"))
filenames_sdi = sorted(set(all_filenames) - set(filenames_adi))

fig, axes = pro.subplots(width="3.5in", height="2.5in")

adi_dfs = [pd.read_csv(f) for f in filenames_adi]
sdi_dfs = [pd.read_csv(f) for f in filenames_sdi]


plate_scale = 5.9e-3  # mas / px
iwa_px = 58e-3 / plate_scale

labels = ("F610", "F670", "F720", "F760")
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, 4))


def plot_cc(dataframe, ax, **kwargs):
    cc = dataframe.query(f"distance >= {iwa_px}")
    rad_arc = cc["distance"] * plate_scale
    contr = cc["contrast_corr"]
    ax.plot(rad_arc.values, contr.values, **kwargs,)

for i in range(len(labels)):
    plot_cc(adi_dfs[i], axes[0], label=labels[i], c=cycle[i], lw=1)

plot_cc(sdi_dfs[1], axes[0], label="ADI+Mean", c="0.2", lw=1)
plot_cc(sdi_dfs[0], axes[0], label="ADI+SDI", c="0.2", ls="--", lw=1)



exp_sep = 135.18
axes[0].scatter([exp_sep * plate_scale], [2e-6], c=cycle[0], marker=".")
axes[0].scatter([exp_sep * plate_scale], [4e-6], c=cycle[1], marker=".")
axes[0].scatter([exp_sep * plate_scale], [1.1e-5], c=cycle[2], marker=".")
axes[0].scatter([exp_sep * plate_scale], [2.0e-5], c=cycle[3], marker=".")

axes[0].text(
    exp_sep * plate_scale + 0.03,
    1.5e-5,
    "HD 1160B",
    color="0.3",
    fontsize=6,
    ha="left",
    va="top",
)

# axes[0].axvline(59e-3, color="0.3", zorder=0, lw=1)


axes[0].legend(ncols=1, frame=False)
axes[0].format(
    xlabel='separation (")',
    ylabel=r"$5\sigma$ contrast",
    yscale="log",
    yformatter="log",
    xlim=(0, 1.4),
)
# pro.show()
fig.savefig(
    paths.figures / "20230711_HD1160_contrast_curve.pdf",
    dpi=300,
)
