import paths
from .utils_psf_fitting import fit_moffat, phot_from_model
from astropy.io import fits
import proplot as pro
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp
import tqdm.auto as tqdm
from astropy.modeling import models, fitting


pro.rc["legend.fontsize"] = 6
pro.rc["title.size"] = 8
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["cmap"] = "bone"

expected_angle = -4.9
res_elems = np.rad2deg(np.array((614, 670, 721, 761)) * 1e-9 / 7.95) * 3.6e6  # masi


def get_initial_offsets(header, angle=expected_angle):
    nangs = 4
    angles = angle + np.arange(nangs) * 90
    seps = header["X_GRDSEP"] * res_elems / 5.6
    offsets = np.zeros((nangs, len(res_elems), 2))
    for i in range(4):
        for j in range(4):
            sint = np.sin(np.deg2rad(angles[j]))
            cost = np.cos(np.deg2rad(angles[j]))
            offsets[i, j] = seps[i] * cost, seps[i] * sint
    return offsets


tbl = pd.read_csv(paths.data / "20240301_astrogrid" / "header_table.csv")
tbl["data"] = tbl["path"].apply(lambda p: fits.getdata(p))
tbl["err"] = tbl["path"].apply(lambda p: fits.getdata(p, ext=("ERR", 1)))

ref_idxs = tbl["X_GRDST"] == "OFF"
reference_images = tbl.loc[ref_idxs, "data"].iloc[0]
reference_errs = tbl.loc[ref_idxs, "err"].iloc[0]
tbl["diff"] = tbl["data"].apply(lambda d: d - reference_images)
for _, row in tbl.iterrows():
    p = row["path"].replace(".fits", "_diff.fits")
    fits.writeto(p, row["diff"], overwrite=True)

tbl.sort_values(["X_GRDST", "X_GRDSEP", "X_GRDAMP"], inplace=True)


subtbl = tbl.loc[tbl["X_GRDST"] != "OFF"].copy(deep=True)

## Step 1: determine centroids from brightest frames
moff_models = {}
moff_off_models = {}
for gk, group in tqdm.tqdm(subtbl.groupby("X_GRDSEP"), total=3):
    bright_sub = group.query("X_GRDAMP == X_GRDAMP.max()")
    for _, row in bright_sub.iterrows():
        data = row["data"]
        diff = row["diff"]
        err = row["err"]

        cy, cx = np.array(data.shape[-2:]) / 2 - 0.5
        offsets = get_initial_offsets(row)
        centroids = offsets + np.array((cx, cy))
        ctr_models = []
        spot_models = []
        for wl_idx in range(4):
            # get fit of central star
            frame = data[wl_idx]
            frame_err = err[wl_idx]
            diff_frame = diff[wl_idx]
            ref_err = reference_errs[wl_idx]
            diff_err = np.hypot(ref_err, frame_err)
            # fit PSF to center of current frame
            center_star = fit_moffat(frame, frame_err, (cx, cy))
            ctr_models.append(center_star)
            off_models = []
            for spot_idx in range(4):
                # fit moffat to DIFF frame
                offset_fit = fit_moffat(frame, frame_err, centroids[wl_idx, spot_idx])
                off_models.append(offset_fit)
            spot_models.append(off_models)

        key = row["X_GRDSEP"]
        moff_models[key] = ctr_models
        moff_off_models[key] = spot_models

rel_fluxes = []
ref_fluxes = []
diff_fluxes = []
for idx, row in tqdm.tqdm(subtbl.iterrows(), total=len(subtbl)):
    data = row["data"]
    diff = row["diff"]
    err = row["err"]
    key = row["X_GRDSEP"]

    ctr_models = moff_models[key]
    satspot_models = moff_off_models[key]

    central_star_flux = []
    ref_star_flux = []
    diff_star_flux = []
    spot_flux = []
    for wl_idx in range(4):
        # get fit of central star
        ref_frame = reference_images[wl_idx]
        ref_err = reference_errs[wl_idx]
        frame = data[wl_idx]
        frame_err = err[wl_idx]
        diff_frame = diff[wl_idx]
        diff_err = np.hypot(ref_err, frame_err)
        # fit PSF to center of current frame
        center_star = ctr_models[wl_idx]
        flux = phot_from_model(frame, frame_err, center_star)
        ref_flux = phot_from_model(ref_frame, ref_err, center_star)
        diff_flux = phot_from_model(diff_frame, diff_err, center_star)

        spot_fluxes = []
        for spot_idx in range(4):
            # fit moffat to DIFF frame
            spot_model = satspot_models[wl_idx][spot_idx]
            # measure flix in DIFF frame
            off_flux = phot_from_model(diff_frame, diff_err, spot_model)
            spot_fluxes.append(off_flux)

        central_star_flux.append(flux)
        ref_star_flux.append(ref_flux)
        diff_star_flux.append(diff_flux)
        spot_flux.append(spot_fluxes)

    central_star_flux = np.array(central_star_flux)
    ref_star_flux = np.array(ref_star_flux)
    diff_star_flux = np.array(diff_star_flux)
    spot_flux = np.array(spot_flux)

    rel_fluxes.append(spot_flux.mean(axis=1) / central_star_flux)
    ref_fluxes.append(ref_star_flux)
    diff_fluxes.append(diff_star_flux / ref_star_flux)

subtbl["diffflux"] = diff_fluxes
subtbl["relflux"] = rel_fluxes
subtbl["refflux"] = ref_fluxes
waves = np.array((614, 670, 721, 761))  # nm

tbl_save = subtbl[["X_GRDAMP", "X_GRDSEP"]].copy()
tbl_save["F610_diff"] = unp.nominal_values(np.array(diff_fluxes)[:, 0])
tbl_save["F610_diff_err"] = unp.std_devs(np.array(diff_fluxes)[:, 0])
tbl_save["F670_diff"] = unp.nominal_values(np.array(diff_fluxes)[:, 1])
tbl_save["F670_diff_err"] = unp.std_devs(np.array(diff_fluxes)[:, 1])
tbl_save["F720_diff"] = unp.nominal_values(np.array(diff_fluxes)[:, 2])
tbl_save["F720_diff_err"] = unp.std_devs(np.array(diff_fluxes)[:, 2])
tbl_save["F760_diff"] = unp.nominal_values(np.array(diff_fluxes)[:, 3])
tbl_save["F760_diff_err"] = unp.std_devs(np.array(diff_fluxes)[:, 3])

tbl_save.to_csv("astrogrid_diff_photometry.csv")

def phot_model(amp, wave, c=1, c1=0):
    opd = amp / wave
    return c * opd**2 + c1 * opd


def phot_deriv(amp, wave, c=1, c1=0):
    opd = amp / wave
    return [opd**2, opd]


PhotModel = models.custom_model(phot_model, fit_deriv=phot_deriv)

fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)

fig, axes = pro.subplots(nrows=3, width="3.5in", height="4in", hspace=0.5, sharey=1)


cycle = pro.Colormap("boreal_r")(np.linspace(0.3, 0.7, 7))
colors = dict(zip((0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07), cycle))

test_waves = np.linspace(600, 775, 1000)

model_fits_pred = {}
for key, group in subtbl.groupby("X_GRDSEP"):
    wvs, aps = np.meshgrid(waves, group["X_GRDAMP"] * 1e3)
    init_model = PhotModel(c=1)
    z = np.array([unp.nominal_values(a) for a in group["relflux"]])
    z_err = np.array([unp.std_devs(a) for a in group["relflux"]])
    fit_model = fitter(init_model, aps, wvs, z, weights=1 / z_err, maxiter=1000)
    chi2 = np.sum((fit_model(aps, wvs) - z) ** 2)
    stds = np.sqrt(fit_model.cov_matrix.cov_matrix.diagonal())
    print(
        f"{fit_model.c.value:.03f}+-{stds[0]:.03f}*opd^2 + {fit_model.c1.value:.03f}+-{stds[1]:.03f}*opd (chi2={chi2})"
    )
    c_samps = np.random.normal(loc=fit_model.c.value, scale=stds[0], size=1000)
    c1_samps = np.random.normal(loc=fit_model.c1.value, scale=stds[1], size=1000)
    pred_values = []
    test_wvs, test_aps = np.meshgrid(test_waves, group["X_GRDAMP"] * 1e3)
    for i in range(1000):
        model = PhotModel(c=c_samps[i], c1=c1_samps[i])
        pred_values.append(model(test_aps, test_wvs))
    model_fits_pred[key] = unp.uarray(
        np.mean(pred_values, axis=0), np.std(pred_values, axis=0)
    )


for i, (key, group) in enumerate(subtbl.groupby("X_GRDSEP")):
    for j, (_, row) in enumerate(group.iterrows()):
        axes[i].scatter(
            waves,
            unp.nominal_values(row["relflux"]),
            c=colors[row["X_GRDAMP"]],
            label=f"{row['X_GRDAMP']*1e3:3.0f} nm",
        )

        ys = model_fits_pred[key][j]

        (row["X_GRDAMP"], test_waves / 1e3)
        axes[i].plot(
            test_waves,
            unp.nominal_values(ys),
            shadedata=unp.std_devs(ys),
            c=colors[row["X_GRDAMP"]],
            alpha=0.7,
            lw=1,
        )
    axes[i].set_ylim(
        0.1 * unp.nominal_values(row["relflux"]).min(),
        4 * unp.nominal_values(row["relflux"]).max(),
    )
    axes[i].format(title=str(key) + r" $\lambda/D$", titleloc="ur")
    axes[i].legend(ncols=3, loc="ul", frame=False)

axes[2].set_ylim(8e-4, 2e-2)
xlim = axes[0].get_xlim()
axes.format(
    xlabel=r"$\lambda$ (nm)",
    ylabel="relative flux",
    yscale="log",
    yformatter="log",
    xlim=xlim,
)
axes[:-1].format(xtickloc="none")


fig.savefig(paths.figures / "astrogrid_photometry.pdf", dpi=300)
