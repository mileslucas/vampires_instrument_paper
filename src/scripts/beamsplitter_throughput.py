import paths
import pandas as pd
import numpy as np
from uncertainties import unumpy

df_open = pd.read_csv(
    paths.data / "20240401_bs_throughput" / "open_throughput_table.csv"
)
df_pbs = pd.read_csv(paths.data / "20240401_bs_throughput" / "pbs_throughput_table.csv")
df_npbs = pd.read_csv(
    paths.data / "20240401_bs_throughput" / "npbs_throughput_table.csv"
)

# sort
for df in (df_open, df_pbs, df_npbs):
    df.sort_values("X_POLARP", inplace=True)

groups = df.groupby(["U_BS", "U_CAMERA"])

# get the flux without a beamsplitter
total_group = df_open
lp_angles = total_group["X_POLARP"].values
total_flux = unumpy.uarray(total_group["PHOTF"].values, total_group["PHOTE"].values)

# and now get flux with the beamsplitters and add together
pbs_groups = df_pbs.groupby("U_CAMERA")
pbs_flux_cam1 = unumpy.uarray(
    pbs_groups.get_group(1)["PHOTF"].values, pbs_groups.get_group(1)["PHOTE"].values
)
pbs_flux_cam2 = unumpy.uarray(
    pbs_groups.get_group(2)["PHOTF"].values, pbs_groups.get_group(2)["PHOTE"].values
)

pbs_relflux_cam1 = pbs_flux_cam1 / total_flux
pbs_relflux_cam2 = pbs_flux_cam2 / total_flux

print(f"PBS Cam 1: {pbs_relflux_cam1.mean()}")
print(f"PBS Cam 2: {pbs_relflux_cam2.mean()}")
print(f"PBS Total: {(pbs_relflux_cam1 + pbs_relflux_cam2).mean()}")

# and now get flux with the beamsplitters and add together
npbs_groups = df_npbs.groupby("U_CAMERA")
npbs_flux_cam1 = unumpy.uarray(
    npbs_groups.get_group(1)["PHOTF"].values, npbs_groups.get_group(1)["PHOTE"].values
)
npbs_flux_cam2 = unumpy.uarray(
    npbs_groups.get_group(2)["PHOTF"].values, npbs_groups.get_group(2)["PHOTE"].values
)

npbs_relflux_cam1 = npbs_flux_cam1 / total_flux
npbs_relflux_cam2 = npbs_flux_cam2 / total_flux

print(f"NPBS Cam 1: {npbs_relflux_cam1.mean()}")
print(f"NPBS Cam 2: {npbs_relflux_cam2.mean()}")
print(f"NPBS Total: {(npbs_relflux_cam1 + npbs_relflux_cam2).mean()}")
