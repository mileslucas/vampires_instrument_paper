import paths
from astropy.io import fits
import proplot as pro
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
import tqdm.auto as tqdm

from skimage import measure
from skimage import morphology
from skimage import feature

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "mono_r"
pro.rc["axes.grid"] = False

masks_name = {
    "Open": "Pupil",
    "PupilRef": "LyotStop-S",
    "LyotOpt": "LyotStop-M",
    "LyotStop": "LyotStop-L",
}

data_dict = {}
for path in (paths.data / "pupil_flats" / "vcam1").glob("*.fits"):
    data, hdr = fits.getdata(path, header=True)
    key = hdr["U_MASK"]
    if key not in masks_name:
        continue
    data_dict[key] = np.nan_to_num(np.squeeze(data))


threshold = 1e4
bin_dict = {k: (d > threshold).astype(int) for k, d in data_dict.items()}


def fit_circle(image, name):
    hull = morphology.convex_hull_image(image)
    edges = feature.canny(hull)
    coords = np.column_stack(np.nonzero(edges))
    model, inliers = measure.ransac(
        coords,
        measure.CircleModel,
        min_samples=500,
        residual_threshold=1,
        max_trials=500,
    )

    # frame_to_show = image + hull
    # plt.subplots()
    # plt.imshow(frame_to_show, origin="lower")
    # plt.gca().add_patch(plt.Circle((model.params[1], model.params[0]), model.params[2], ec="r", lw=2, fill=False))
    # plt.title(name, loc="left")
    # plt.show()

    return model.params


mask_diams = {"Open": 7.95, "PupilRef": 7.06, "LyotOpt": 6.99, "LyotStop": 6.33}
rows = []


for k in tqdm.tqdm(masks_name):
    ratio = np.nansum(bin_dict[k]) / np.nansum(bin_dict["Open"])

    cy, cx, r = fit_circle(bin_dict[k], masks_name[k])
    rows.append(
        {"name": k, "cx": cx, "cy": cy, "r": r, "T_geom": ratio, "diam": mask_diams[k]}
    )
pup_df = pd.DataFrame(rows)

pup_df["plate_scale"] = pup_df["diam"] / 2 / pup_df["r"] * 1e3

print(pup_df)
print(pup_df.iloc[1:]["plate_scale"].describe())
pup_df.to_csv(paths.data / "pupil_measurements.csv")
# plate_scale = 6.03 # mas / px

fig, axes = pro.subplots(nrows=2, ncols=2, width="3.5in", space=0.25)

for ax, key in zip(axes, masks_name):
    frame = bin_dict[key]
    cx = pup_df.loc[pup_df["name"] == key, "cx"]
    cy = pup_df.loc[pup_df["name"] == key, "cy"]
    cutout = Cutout2D(frame.astype(float), (cx, cy), 850, mode="partial")
    side_length = np.array(cutout.shape)
    ext = (
        -side_length[1] / 2,
        side_length[1] / 2,
        -side_length[0] / 2,
        side_length[0] / 2,
    )
    ax.imshow(
        cutout.data,
        extent=ext,
        cmap="mono_r",  # , norm=simple_norm(cutout.data, "asinh")
    )
    ax.text(
        0.03,
        0.97,
        masks_name[key],
        c="w",
        va="top",
        ha="left",
        fontsize=7,
        transform="axes",
    )

axes.format(xticks=[], yticks=[])
axes[:, 1].format(ytickloc="none")

# save output
fig.savefig(paths.figures / "pupil_masks.pdf", dpi=300)
