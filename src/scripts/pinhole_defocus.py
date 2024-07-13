import paths
import proplot as pro
import numpy as np
import pandas as pd
from astropy.modeling import models, fitting
from uncertainties import ufloat

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["cycle"] = "ggplot"

table = pd.read_csv(paths.data / "pinhole_defocus.csv", index_col=0)
focus_posns = table.index
X = focus_posns - focus_posns[15]
ave_vals = np.nanmean(table.values, 1)
std_vals = np.nanstd(table.values, 1)

fig, axes = pro.subplots(
    width="3.5in",
    height="2.5in",
)



fitter = fitting.LinearLSQFitter(calc_uncertainties=True)
mod = models.Polynomial1D(1, domain=(X.min(), X.max()))
mod.c0.fixed = True
res = fitter(mod, X, ave_vals)#, weights=1/std_vals)
stds = np.sqrt(res.cov_matrix.cov_matrix.diagonal())
res_slope = ufloat(res.c1.value, stds[0])
print(res_slope)

axes[0].scatter(
    X,
    ave_vals * 100,
    label="data",
)
test_X = np.linspace(X.min() - 0.1, X.max() + 0.1, 100)
axes[0].plot(
    test_X,
    res(test_X) * 100,
    c="C0",
    label="model"
)

axes[0].format(
    ylabel="Change in platescale (%)",
    xlabel="Lens defocus (mm)"
)
axes[0].legend(ncols=1, frame=False, )

axes.format(
    grid=True
)

# save output
fig.savefig(paths.figures / "pinhole_defocus.pdf", dpi=300)
