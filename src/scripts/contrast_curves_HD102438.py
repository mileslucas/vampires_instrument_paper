import paths
import proplot as pro
import numpy as np
import pandas as pd

pro.rc["legend.fontsize"] = 8
pro.rc["font.size"] = 9
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cycle"] = "ggplot"

all_filenames = (paths.data / "HD102438_contrast").glob("*.csv")
filenames_adi = sorted((paths.data / "HD102438_contrast").glob("*F*.csv"))
filenames_sdi = sorted(set(all_filenames) - set(filenames_adi))
fig, axes = pro.subplots(width="3.5in", height="2.5in")

adi_dfs = [pd.read_csv(f) for f in filenames_adi]
sdi_dfs = [pd.read_csv(f) for f in filenames_sdi]


plate_scale = 5.9e-3  # arc / px
iwa = 0.105  # arc
iwa_px = iwa / plate_scale


labels = ("F610", "F670", "F720", "F760")
cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, 4))


def plot_cc(dataframe, ax, **kwargs):
    cc = dataframe.query(f"distance >= {iwa_px}")
    rad_arc = cc["distance"] * plate_scale
    contr = cc["contrast_corr"]
    ax.plot(
        rad_arc.values,
        contr.values,
        **kwargs,
    )


for i in range(len(labels)):
    plot_cc(adi_dfs[i], axes[0], label=labels[i], c=cycle[i], lw=1)

plot_cc(sdi_dfs[1], axes[0], label="ADI+Mean", c="0.2", lw=1)
plot_cc(sdi_dfs[0], axes[0], label="ADI+SDI", c="0.2", ls="--", lw=1)

# axes[0].axvline(105e-3, color="0.3", zorder=0, lw=1)
# axes[0].text(
#     0.08,
#     0.03,
#     "IWA",
#     color="0.3",
#     fontsize=7,
#     ha="right",
#     va="bottom",
#     transform="axes",
# )

axes[0].legend(ncols=1, frame=False)
axes[0].format(
    xlabel='separation (")',
    ylabel=r"$5\sigma$ contrast",
    yscale="log",
    yformatter="log",
    xlim=(0, 1.4),
    ylim=(2e-7, 2e-2),
)
fig.savefig(
    paths.figures / "20230629_HD102438_contrast_curve.pdf",
    dpi=300,
)
