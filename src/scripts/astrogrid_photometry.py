import paths
from astropy.io import fits
import proplot as pro
import numpy as np
from photutils.centroids import centroid_quadratic
import pandas as pd
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values
import sep
import tqdm.auto as tqdm



pro.rc["legend.fontsize"] = 6
pro.rc["title.size"] = 8
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8



tbl = pd.read_csv(paths.data / "20240221_astrogrid" / "header_table.csv")
tbl["centroid_path"] = tbl["path"].apply(lambda p: p.replace(".fits", "_centroids.csv"))
tbl["data"] = tbl["path"].apply(lambda p: fits.getdata(p))
tbl["err"] = tbl["path"].apply(lambda p: fits.getdata(p, ext=("ERR", 1)))
tbl.sort_values(["X_GRDST", "X_GRDSEP", "X_GRDAMP"], inplace=True)


def aperture_phot(frame, frame_err, x, y, rad=12):
    _frame = np.nan_to_num(frame.astype("=f4"))
    _err = np.nan_to_num(frame_err.astype("=f4"))
    flux, fluxerr, _ = sep.sum_circle(_frame, (x,), (y,), rad, err=_err)
    return ufloat(flux[0], fluxerr[0])


def refined_centroid(cube, centroid):
    cxs = []
    cys = []
    for _, row in centroid.iterrows():
        frame = cube[row["field"]]
        cx, cy = centroid_quadratic(
            frame, xpeak=row["x"], ypeak=row["y"], fit_boxsize=7, search_boxsize=19
        )
        cxs.append(cx)
        cys.append(cy)
    centroid["rx"] = cxs
    centroid["ry"] = cys
    return centroid


def get_phot_sums(cube, cube_err, centroid):
    photf = []
    for _, row in centroid.iterrows():
        frame = cube[row["field"]]
        frame_err = cube_err[row["field"]]
        flux = aperture_phot(frame, frame_err, row["rx"], row["ry"])
        photf.append(flux)
    centroid["photf"] = photf
    return centroid


rel_fluxes = []
for _, row in tqdm.tqdm(tbl.iterrows(), total=len(tbl)):
    data = row["data"]
    err = row["err"]
    centroid_df = pd.read_csv(row["centroid_path"])
    refined_df = refined_centroid(data, centroid_df)

    photf_df = get_phot_sums(data, err, refined_df)

    groups = photf_df.groupby("psf")
    rel_flux_g1 = (
        groups.get_group("G1")["photf"].values / groups.get_group("PSF")["photf"].values
    )
    rel_flux_g2 = (
        groups.get_group("G2")["photf"].values / groups.get_group("PSF")["photf"].values
    )
    rel_flux_g3 = (
        groups.get_group("G3")["photf"].values / groups.get_group("PSF")["photf"].values
    )
    rel_flux_g4 = (
        groups.get_group("G4")["photf"].values / groups.get_group("PSF")["photf"].values
    )
    norm_flux = (rel_flux_g1, rel_flux_g2, rel_flux_g3, rel_flux_g4)
    rel_flux = np.mean(norm_flux, axis=0)
    # print(rel_flux)
    rel_fluxes.append(rel_flux)

tbl["relflux"] = rel_fluxes
waves = np.array((614, 670, 721, 761)) * 1e-3  # micron
tbl["coeff"] = (
    tbl["relflux"].apply(lambda f: np.mean(f * waves**2)) / (tbl["X_GRDAMP"] ** 2)
)


fig, axes = pro.subplots(nrows=3, width="3.5in", height="4in", hspace=0.5, sharey=1)


cycle = pro.Colormap("boreal_r")(np.linspace(0.3, 0.7, 5))
colors = dict(zip((0.025, 0.05, 0.075, 0.1, 0.125), cycle))

sub_tbl = tbl.query("X_GRDST != 'OFF'")

coeffs = {}
for key, group in tbl.groupby(["X_GRDST", "X_GRDSEP"]):
    if key[0] != "OFF":
        coeffs[key] = np.mean(group["coeff"].values)
        coeffs[key] = np.mean(group["coeff"].values)
        print(f"{key}: {coeffs[key]}")


test_waves = np.linspace(600, 775, 1000)
for i, (key, group) in enumerate(sub_tbl.groupby("X_GRDSEP")):
    for _, row in group.iterrows():
        axes[i].scatter(
            waves * 1e3, nominal_values(row["relflux"]), c=colors[row["X_GRDAMP"]]
        )

        axes[i].plot(
            test_waves,
            (row["X_GRDAMP"] / (test_waves * 1e-3)) ** 2 * nominal_values(coeffs[("XYgrid", key)]),
            c=colors[row["X_GRDAMP"]],
            alpha=0.7,
            lw=1,
        )
    axes[i].format(title=str(key) + r" $\lambda/D$", titleloc="ur")

xlim = axes[0].get_xlim()
for amp, color in colors.items():
    axes[0].plot(-1, 1, marker=".", c=color, label=f"{amp*1e3:3.0f} nm")
axes[0].legend(ncols=3, loc="ll", frame=False)

axes.format(
    xlabel=r"$\lambda$ (nm)",
    ylabel="relative flux",
    yscale="log",
    yformatter="log",
    ylim=(1.5e-3, 1),
    xlim=xlim
)
axes[-1].format(ylim=(1e-3, 1e-1))
axes[:-1].format(xtickloc="none")


fig.savefig(paths.figures / "astrogrid_photometry.pdf", dpi=300)
