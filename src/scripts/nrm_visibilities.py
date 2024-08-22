import paths
import proplot as pro
import pickle

pro.rc["axes.grid"] = True
pro.rc["font.size"] = 9
pro.rc["title.size"] = 8
pro.rc["legend.fontsize"] = 7

data_path = paths.data / "nrm_visibilities.pkl"
with data_path.open("rb") as fh:
    data = pickle.load(fh)

fig, axes = pro.subplots(nrows=2, width="3.5in", refheight="2in", hspace=0.75)


azimuth = data["xdata"]
im = axes[0].scatter(
    azimuth, data["qvis"], c=data["zdata"], cmap="jet", marker="x", zorder=300
)
axes[0].colorbar(im, label="Baseline length (m)", labelsize=9)
axes[0].text(0.03, 0.03, "Stokes Q", c="0.3", fontsize=11, transform="axes")


im = axes[1].scatter(
    azimuth, data["uvis"], c=data["zdata"], cmap="jet", marker="x", zorder=300
)
axes[1].colorbar(im, label="Baseline length (m)", labelsize=9)
axes[1].text(0.03, 0.03, "Stokes U", c="0.3", fontsize=11, transform="axes")


for ax in axes:
    ax.axhline(1, c="0.3", zorder=1)

axes[0].format(xtickloc="none")

axes.format(
    xlabel="Azimuth angle (rad)", ylabel="Differential visibilities", xlocator=0.5
)

# save output
fig.savefig(paths.figures / "stokes_visibilities.pdf", dpi=300)
