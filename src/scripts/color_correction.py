from scexao_etc.filters import (
    load_vampires_filter,
    VAMPIRES_STD_FILTERS,
    FILTERS,
)
from scexao_etc.models import PICKLES_MAP, load_pickles, color_correction
import paths
import proplot as pro
import numpy as np
import pandas as pd
import re
from matplotlib import ticker

pro.rc["axes.grid"] = True
pro.rc["font.size"] = 8
pro.rc["title.size"] = 8
pro.rc["legend.fontsize"] = 7


def color_corr(sptype, src_filter, vamp_filter):
    model = load_pickles(sptype)
    src_filter = FILTERS[src_filter]
    vamp_filter = load_vampires_filter(vamp_filter)
    return color_correction(model, src_filter, vamp_filter)


rows = []
for sptype in PICKLES_MAP:
    if not re.search(r"^\w\dV", sptype):
        continue
    colors = {f: color_corr(sptype, "V", f) for f in VAMPIRES_STD_FILTERS}
    colors["sptype"] = sptype[:2]
    rows.append(colors)

color_df = pd.DataFrame(rows)

fig, axes = pro.subplots(nrows=2, width="3.5in", height="3.1in", space=0.5, sharey=1)

cycle = pro.Colormap("fire")(np.linspace(0.4, 0.8, len(VAMPIRES_STD_FILTERS) - 1))
cmap = dict(zip(("625-50", "675-50", "725-50", "750-50", "775-50"), cycle))

axes[0].plot(
    color_df["sptype"], color_df["625-50"], c=cmap["625-50"], label="V - 625-50"
)
axes[0].plot(
    color_df["sptype"], color_df["675-50"], c=cmap["675-50"], label="V - 675-50"
)
axes[0].plot(
    color_df["sptype"], color_df["725-50"], c=cmap["725-50"], label="V - 725-50"
)
axes[0].plot(
    color_df["sptype"], color_df["750-50"], c=cmap["750-50"], label="V - 750-50"
)
axes[0].plot(
    color_df["sptype"], color_df["775-50"], c=cmap["775-50"], label="V - 775-50"
)
axes[0].plot(color_df["sptype"], color_df["Open"], c="0.3", label="V - Open")

axes[0].legend(ncols=2, order="F")


rows = []
for sptype in PICKLES_MAP:
    if not re.search(r"^\w\dV", sptype):
        continue
    colors = {f: color_corr(sptype, "R", f) for f in VAMPIRES_STD_FILTERS}
    colors["sptype"] = sptype[:2]
    rows.append(colors)

color_df = pd.DataFrame(rows)

axes[1].plot(
    color_df["sptype"], color_df["625-50"], c=cmap["625-50"], label="R - 625-50"
)
axes[1].plot(
    color_df["sptype"], color_df["675-50"], c=cmap["675-50"], label="R - 675-50"
)
axes[1].plot(
    color_df["sptype"], color_df["725-50"], c=cmap["725-50"], label="R - 725-50"
)
axes[1].plot(
    color_df["sptype"], color_df["750-50"], c=cmap["750-50"], label="R - 750-50"
)
axes[1].plot(
    color_df["sptype"], color_df["775-50"], c=cmap["775-50"], label="R - 775-50"
)
axes[1].plot(color_df["sptype"], color_df["Open"], c="0.3", label="R - Open")

axes[1].legend(ncols=2, order="F")

xticks = axes[1].get_xticks()
axes[1].set_xticks(xticks[1::2])
axes[1].set_xticklabels(color_df["sptype"][1::2], rotation=90)

axes.format(
    xlabel="spectral type",
    ylabel="color correction (mag)",
)
axes[0].format(xtickloc="none")
axes[1].format(
    ylim=(None, 2.5), ylocator=ticker.MaxNLocator(nbins="auto", prune="upper")
)
# save output
fig.savefig(paths.figures / "color_correction.pdf", dpi=300)
