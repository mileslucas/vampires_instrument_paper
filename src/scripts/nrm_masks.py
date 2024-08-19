import paths
from astropy.io import fits
import proplot as pro
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import tqdm.auto as tqdm
from pathlib import Path
from dataclasses import dataclass
from numpy.typing import NDArray
from matplotlib import patches, ticker
import hcipy as hp

pro.rc["legend.fontsize"] = 6
pro.rc["font.size"] = 8
pro.rc["legend.title_fontsize"] = 8
pro.rc["image.origin"] = "lower"
pro.rc["cmap"] = "mono_r"
pro.rc["axes.grid"] = False
pro.rc["axes.facecolor"] = "k"

@dataclass
class NRMMask:
    hole_rad: float
    coords: NDArray

    def make_array(self, npix, diameter):
        pupil_grid = hp.make_pupil_grid((npix, npix), diameter=diameter)
        pupil = hp.Field(np.zeros(pupil_grid.shape).ravel(), pupil_grid)
        for coord in self.coords:
            hole = hp.make_circular_aperture(mask.hole_rad * 2, center=coord)
            pupil += hp.evaluate_supersampled(hole, pupil_grid, 16)
        return pupil.shaped
    
from typing import Iterable

@dataclass
class AnnMask:
    inner_rad: float
    outer_rad: float
    theta_starts: Iterable[float]
    theta_ends: Iterable[float]

    def make_array(self, npix, diameter):
        pupil_grid = hp.make_pupil_grid((npix, npix), diameter=diameter)
        pol_grid = pupil_grid.as_('polar')
        pupil = hp.Field(np.zeros(pupil_grid.shape).ravel(), pupil_grid)
        outer = hp.make_circular_aperture(self.outer_rad * 2)
        inner = hp.make_circular_aperture(self.inner_rad * 2)
        ring = hp.evaluate_supersampled(outer, pupil_grid, 16) - \
                hp.evaluate_supersampled(inner, pupil_grid, 16)
        
        theta_grid = np.rad2deg(pol_grid.theta)
        theta_grid[theta_grid < 0] += 360
        for t0, t1 in zip(self.theta_starts, self.theta_ends):
            if t0 < 0:
                t0 += 360
                mask = (theta_grid >= t0) | (theta_grid <= t1)
            else:
                mask = (theta_grid >= t0) & (theta_grid <= t1)
            ring[mask] = 0
        pupil += ring

        return pupil.shaped

def get_data_from_file(filename):
    full_text = Path(filename).read_text()
    lines = full_text.split("\n")
    start_idx = 0
    for line in lines:
        if "---COORDINATES at M1" in line:
            break
        start_idx +=1
    # we know data starts 2 lines after coordinate line
    chunk = lines[start_idx + 2:]
    # find hole radius line
    rad_idx = 0
    for line in chunk:
        if "Hole Radius:" in line:
            break
        rad_idx += 1
    hole_rad_line = chunk[rad_idx]
    hole_rad = float(hole_rad_line.split()[-1]) # m
    coords = np.array([
        list(map(float, line.split())) for line in chunk[:rad_idx - 1]
    ]) # x, y (m)
    return NRMMask(hole_rad=hole_rad, coords=coords)


def get_ann_from_file(filename):
    full_text = Path(filename).read_text()
    lines = full_text.split("\n")
    start_idx = 0
    for line in lines:
        if "Inner radius" in line:
            inner_rad = float(line.split()[3])
        if "Outer radius" in line:
            outer_rad = float(line.split()[3])
        if "Segment" in line:
            break
        start_idx +=1
    # we know data starts 2 lines after coordinate line
    chunk = lines[start_idx:]
    theta_starts = []
    theta_ends = []
    for line in chunk:
        if "Segment" not in line:
            continue
        tokens = line.split()
        theta_starts.append(float(tokens[2]))
        theta_ends.append(float(tokens[5]))

    return AnnMask(inner_rad=inner_rad, outer_rad=outer_rad, theta_starts=theta_starts, theta_ends=theta_ends)

def plot_nrm_mask(mask: NRMMask, ax, name):
    for coord in mask.coords:
        patch = patches.Circle(coord, radius=mask.hole_rad, color="w", fill=True, zorder=350)
        ax.add_patch(patch)
    ax.text(0.03, 0.96, name, ha="left", va="top", c="w", fontsize=7, transform="axes")
    return ax


def plot_ann_mask(mask: AnnMask, ax, name):
    diameter = 10
    npix = int(((512 * diameter / 7.95) // 2) * 2)
    arr = mask.make_array(npix, diameter)
    arr[arr== 0] = np.nan
    ext = (-diameter/2, diameter/2, -diameter/2, diameter/2)
    ax.imshow(arr, cmap="mono_r", extent=ext, zorder=350)
    ax.text(0.03, 0.96, name, ha="left", va="top", c="w", fontsize=7, transform="axes")
    return ax

def plot_interferogram(mask, ax, name):
    diameter = 7.95 * 2
    npix = int(((256 * diameter / 7.95) // 2) * 2)
    base_pupil = mask.make_array(npix=npix, diameter=diameter)


    interf = np.abs(np.fft.fft2(base_pupil))**2
    psd = np.abs(np.fft.fftshift(np.fft.fft2(interf)))**2
    half_width = diameter / 2
    ext = (-half_width, half_width,-half_width, half_width)

    psd_plot = np.clip(psd, psd.min(), np.percentile(psd, 95))

    ax.imshow(psd_plot, cmap="mono_r", extent=ext)
    ax.text(0.03, 0.96, name, ha="left", va="top", c="w", fontsize=7, transform="axes")


data_dict = {
    "SAM 7": paths.data / "nrm_masks" / "vampires_g7_reflective_ThickSpiders.txt",
    "SAM 9": paths.data / "nrm_masks" / "vampires_g9_reflective_ThickSpiders.txt",
    "SAM 18": paths.data / "nrm_masks" / "vampires_g18_reflective_ThickSpiders_NUDGED.txt",
    "SAM Ann": paths.data / "nrm_masks" / "annulus_reflectivebench_ThickSpiders.txt"
}

fig, axes = pro.subplots(nrows=2, ncols=4, width="7in", wspace=0.25, sharex=1, spany=False)


for i, key in tqdm.tqdm(enumerate(data_dict), total=len(data_dict)):
    if key == "SAM Ann":
        mask = get_ann_from_file(data_dict[key])
        plot_ann_mask(mask, axes[0, i], name=key)
    else:
        mask = get_data_from_file(data_dict[key])
        plot_nrm_mask(mask, axes[0, i], name=key)
    outer_circ = patches.Circle((0, 0), radius=7.95/2, ec="w", fc="0.4", lw=1, fill=True, zorder=300)
    axes[0, i].add_patch(outer_circ)
    inner_circ = patches.Circle((0, 0), radius=2.337/2, ec="w", fc="k",  lw=1, fill=True, zorder=400)
    axes[0, i].add_patch(inner_circ)
    axes[0, i].scatter([0], [0], c="w", marker="+", ms=50, lw=1, zorder=500)

    plot_interferogram(mask, axes[1, i], name=key)

for ax in axes:
    ax.format(
        ylocator=ticker.MaxNLocator(5, prune="both"),
        xlocator=ticker.MaxNLocator(5, prune="both")
    )

axes[0, 0].format(ylabel="Pupil position (m)")
axes[:, 1:].format(ytickloc="none")

axes[1, 0].format(ylabel="Fourier coverage (m)")
axes[1, :].format(xlabel="Fourier coverage (m)")

axes[0, :].format(
    xlim=(-4.5, 4.5),
    ylim=(-4.5, 4.5),
)
axes[1, :].format(
    xlim=(-8.5, 8.5),
    ylim=(-8.5, 8.5)
)


# save output
fig.savefig(paths.figures / "nrm_masks.pdf", dpi=300)
