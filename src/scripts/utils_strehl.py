
import numpy as np
import urllib
import sep
from astropy.nddata import Cutout2D
from skimage.registration import phase_cross_correlation

import hcipy as hp
import numpy as np
from astropy.io import fits

from astropy.utils.data import download_file
from synphot import SpectralElement


def find_peak(image, xc, yc, boxsize, oversamp=8):
    """
    usage: peak = find_peak(image, xc, yc, boxsize)
    finds the subpixel peak of an image

    image: an image of a point source for which we would like to find the peak
    xc, yc: approximate coordinate of the point source
    boxsize: region in which most of the flux is contained (typically 20)
    oversamp: how many times to oversample the image in the FFT interpolation in order to find the peak

    :return: peak of the oversampled image

    Marcos van Dam, October 2022, translated from IDL code of the same name
    """

    boxhalf = np.ceil(boxsize / 2.0).astype(int)
    boxsize = 2 * boxhalf
    ext = np.array(boxsize * oversamp, dtype=int)

    # need to deconvolve the image by dividing by a sinc in order to "undo" the sampling
    fftsinc = np.zeros(ext)
    fftsinc[0:oversamp] = 1.0

    sinc = (
        boxsize
        * np.fft.fft(fftsinc, norm="forward")
        * np.exp(
            1j * np.pi * (oversamp - 1) * np.roll(np.arange(-ext / 2, ext / 2), int(ext / 2)) / ext
        )
    )
    sinc = sinc.real
    sinc = np.roll(sinc, int(ext / 2))
    sinc = sinc[int(ext / 2) - int(boxsize / 2) : int(ext / 2) + int(boxsize / 2)]
    sinc2d = np.outer(sinc, sinc)

    # define a box around the center of the star
    blx = np.floor(xc - boxhalf).astype(int)
    bly = np.floor(yc - boxhalf).astype(int)

    # make sure that the box is contained by the image
    blx = np.clip(blx, 0, image.shape[0] - boxsize)
    bly = np.clip(bly, 0, image.shape[1] - boxsize)

    # extract the star
    subim = image[blx : blx + boxsize, bly : bly + boxsize]

    # deconvolve the image by dividing by a sinc in order to "undo" the pixelation
    fftim1 = np.fft.fft2(subim, norm="forward")
    shfftim1 = np.roll(fftim1, (-boxhalf, -boxhalf), axis=(1, 0))
    shfftim1 /= sinc2d  # deconvolve

    zpshfftim1 = np.zeros((oversamp * boxsize, oversamp * boxsize), dtype="complex64")
    zpshfftim1[0:boxsize, 0:boxsize] = shfftim1

    zpfftim1 = np.roll(zpshfftim1, (-boxhalf, -boxhalf), axis=(1, 0))
    subimupsamp = np.fft.ifft2(zpfftim1, norm="forward").real

    peak = np.nanmax(subimupsamp)

    return peak


def measure_strehl(image, psf_model, pos=None, phot_rad=0.5, peak_search_rad=0.1, pxscale=5.9):
    ## Step 1: find approximate location of PSF in image

    # If no position given, start at the nan-max
    if pos is None:
        pos = np.unravel_index(np.nanargmax(image), image.shape)
    center = np.array(pos)
    # Now, refine this centroid using cross-correlation
    # this cutout must have same shape as PSF reference (chance for errors here)
    cutout = Cutout2D(image, center[::-1], psf_model.shape)
    assert cutout.data.shape == psf_model.shape

    shift, _, _ = phase_cross_correlation(
        psf_model.astype("=f4"), cutout.data.astype("=f4"), upsample_factor=4
    )
    refined_center = center + shift

    ## Step 2: Calculate peak flux with subsampling and flux
    aper_rad_px = phot_rad / (pxscale * 1e-3)
    image_flux, image_fluxerr, _ = sep.sum_circle(
        image.astype("=f4"),
        (refined_center[1],),
        (refined_center[0],),
        aper_rad_px,
        err=np.sqrt(np.maximum(image, 0)),
    )
    peak_search_rad_px = peak_search_rad / (pxscale * 1e-3)
    image_peak = find_peak(image, refined_center[0], refined_center[1], peak_search_rad_px)

    ## Step 3: Calculate flux of PSF model
    # note: our models are alrady centered
    model_center = np.array(psf_model.shape[-2:]) / 2 - 0.5
    # note: our models have zero background signal
    model_flux, model_fluxerr, _ = sep.sum_circle(
        psf_model.astype("=f4"),
        (model_center[1],),
        (model_center[0],),
        aper_rad_px,
        err=np.sqrt(np.maximum(psf_model, 0)),
    )
    model_peak = find_peak(psf_model, model_center[0], model_center[1], peak_search_rad_px)

    ## Step 4: Calculate Strehl via normalized ratio
    image_norm_peak = image_peak / image_flux[0]
    model_norm_peak = model_peak / model_flux[0]
    strehl = image_norm_peak / model_norm_peak
    return strehl


## constants
PUPIL_DIAMETER = 7.95  # m
OBSTRUCTION_DIAMETER = 5.27 / 17.92 * PUPIL_DIAMETER  # m
INNER_RATIO = OBSTRUCTION_DIAMETER / PUPIL_DIAMETER
SPIDER_WIDTH = 0.1735  # m
SPIDER_OFFSET = 0.639  # m, spider intersection offset
SPIDER_ANGLE = 51.75  # deg
ACTUATOR_SPIDER_WIDTH = 0.089  # m
ACTUATOR_SPIDER_OFFSET = (0.521, -1.045)
ACTUATOR_DIAMETER = 0.632  # m
ACTUATOR_OFFSET = ((1.765, 1.431), (-0.498, -2.331))  # (x, y), m

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def field_combine(field1, field2):
    return lambda grid: field1(grid) * field2(grid)


def create_synth_psf(filt, npix=30, output_directory=None, nwave=7, **kwargs):
    if output_directory is not None:
        outfile = output_directory / f"VAMPIRES_{filt}_synthpsf.fits"
        if outfile.exists():
            psf = fits.getdata(outfile)
            if psf.shape == (npix, npix):
                return psf
    # assume header is fixed already
    pupil_data = generate_pupil(angle=38.49)
    pupil_grid = hp.make_pupil_grid(pupil_data.shape, diameter=PUPIL_DIAMETER)
    pupil_field = hp.Field(pupil_data.ravel(), pupil_grid)
    # create detector grid
    plate_scale = np.deg2rad(5.9 / 1e3 / 60 / 60)  # mas/px -> rad/px
    focal_grid = hp.make_uniform_grid((npix, npix), (plate_scale * npix, plate_scale * npix))
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

    obs_filt = load_vampires_filter(filt)
    waves = obs_filt.waveset
    through = obs_filt.model.lookup_table
    above_50 = np.nonzero(through >= 0.5 * np.nanmax(through))
    waves = np.linspace(waves[above_50].min(), waves[above_50].max(), nwave)

    field_sum = 0
    for wave, through in zip(waves, obs_filt(waves)):
        wf = hp.Wavefront(pupil_field, wave.to("m").value)
        focal_plane = prop(wf).intensity * through.value
        field_sum += focal_plane.shaped
    normed_field = np.flip(field_sum / field_sum.sum(), axis=-2).astype("f4")
    if output_directory is not None:
        fits.writeto(outfile, normed_field, overwrite=True)
    return normed_field

VAMP_FILT_KEY = "1FHGh3tLlDUwATP6smFGz0nk2e0NF14rywTUjFTUT1OY"
VAMP_FILT_NAME = urllib.parse.quote("VAMPIRES Filter Curves")
VAMPIRES_FILTER_URL = f"https://docs.google.com/spreadsheets/d/{VAMP_FILT_KEY}/gviz/tq?tqx=out:csv&sheet={VAMP_FILT_NAME}"


def load_vampires_filter(name: str):
    csv_path = download_file(VAMPIRES_FILTER_URL, cache=True)
    return SpectralElement.from_file(csv_path, wave_unit="nm", include_names=["wave", name])

def generate_pupil(
    n: int = 256,
    outer: float = 1,
    inner: float = INNER_RATIO,
    scale: float = 1,
    angle: float = 0,
    oversample: int = 8,
    spiders: bool = True,
    actuators: bool = True,
):
    pupil_diameter = PUPIL_DIAMETER * outer
    # make grid over full diameter so undersized pupils look undersized
    max_diam = PUPIL_DIAMETER if outer <= 1 else pupil_diameter
    grid = hp.make_pupil_grid(n, diameter=max_diam)

    # This sets us up with M1+M2, just need to add spiders and DM masks
    # fix ratio
    inner_val = inner * PUPIL_DIAMETER
    inner_fixed = inner_val / pupil_diameter
    pupil_field = hp.make_obstructed_circular_aperture(pupil_diameter, inner_fixed)

    # add spiders to field generator
    if spiders:
        spider_width = SPIDER_WIDTH * scale
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))

        # spider in quadrant 1
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET, sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 2
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET, sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 3
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET, -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 4
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET, -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )

    # add actuator masks to field generator
    if actuators:
        # circular masks
        actuator_diameter = ACTUATOR_DIAMETER * scale
        actuator_mask_1 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter, center=ACTUATOR_OFFSET[0])
        )
        pupil_field = field_combine(pupil_field, actuator_mask_1)

        actuator_mask_2 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter, center=ACTUATOR_OFFSET[1])
        )
        pupil_field = field_combine(pupil_field, actuator_mask_2)

        # spider
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))
        actuator_spider_width = ACTUATOR_SPIDER_WIDTH * scale
        actuator_spider = hp.make_spider(
            ACTUATOR_SPIDER_OFFSET,
            (
                ACTUATOR_SPIDER_OFFSET[0] - cost * pupil_diameter,
                ACTUATOR_SPIDER_OFFSET[1] - sint * pupil_diameter,
            ),
            spider_width=actuator_spider_width,
        )
        pupil_field = field_combine(pupil_field, actuator_spider)

    rotated_pupil_field = hp.make_rotated_aperture(pupil_field, np.deg2rad(angle))

    pupil = hp.evaluate_supersampled(rotated_pupil_field, grid, oversample)
    return pupil.shaped

