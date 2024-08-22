import numpy as np
from astropy import modeling
from uncertainties import ufloat
import sep
from astropy.nddata import Cutout2D
import proplot as pro


def aperture_phot(frame, frame_err, x, y, a, b, theta):
    _frame = np.nan_to_num(frame.astype("=f4"))
    _err = np.nan_to_num(frame_err.astype("=f4"))
    flux, fluxerr, _ = sep.sum_ellipse(
        _frame, (x,), (y,), a, b, theta, err=_err, bkgann=(2, 3)
    )
    return ufloat(flux[0], fluxerr[0])


def moffat_fwhm(gamma, alpha):
    return 2 * gamma * np.sqrt(2 ** (1 / alpha) - 1)


def moffat_fwhm_err(gamma, gamma_err, alpha, alpha_err):
    d_gamma = 2 * np.sqrt(2 ** (1 / alpha) - 1)
    d_alpha = (
        -np.log(2)
        * 2 ** (1 / alpha)
        * gamma
        / (alpha**2 * np.sqrt(2 ** (1 / alpha) - 1))
    )
    fwhm_err = np.hypot(d_gamma * gamma_err, d_alpha * alpha_err)
    return fwhm_err


def moffat_gamma(fwhm, alpha):
    return fwhm / (2 * np.sqrt(2 ** (1 / alpha) - 1))


## moffat
class Moffat(modeling.Fittable2DModel):
    x0 = modeling.Parameter()
    y0 = modeling.Parameter()
    gammax = modeling.Parameter(default=1, min=0)
    gammay = modeling.Parameter(default=1, min=0)
    theta = modeling.Parameter(default=0, min=-np.pi / 4, max=np.pi / 4)
    alpha = modeling.Parameter(default=1, min=0)
    amplitude = modeling.Parameter(default=1, min=0)
    background = modeling.Parameter(default=0)

    @property
    def fwhmx(self) -> modeling.Parameter:
        return moffat_fwhm(self.gammax, self.alpha)

    @fwhmx.setter
    def fwhmx(self, fwhmx: float):
        self.gammax = moffat_gamma(fwhmx, self.alpha)

    @property
    def fwhmy(self) -> modeling.Parameter:
        return moffat_fwhm(self.gammay, self.alpha)

    @fwhmy.setter
    def fwhmy(self, fwhmy: float):
        self.gammay = moffat_gamma(fwhmy, self.alpha)

    @staticmethod
    def evaluate(x, y, x0, y0, gammax, gammay, theta, alpha, amplitude, background):
        diffx = x - x0
        diffy = y - y0

        cost = np.cos(theta)
        sint = np.sin(theta)

        a = (cost / gammax) ** 2 + (sint / gammay) ** 2
        b = (sint / gammax) ** 2 + (cost / gammay) ** 2
        c = 2 * sint * cost * (1 / gammax**2 - 1 / gammay**2)

        rad = a * diffx**2 + b * diffy**2 + c * diffx * diffy
        return amplitude / (1 + rad) ** alpha + background

    @staticmethod
    def fit_deriv(x, y, x0, y0, gammax, gammay, theta, alpha, amplitude, background):
        diffx = x - x0
        diffy = y - y0

        cost = np.cos(theta)
        sint = np.sin(theta)
        cos2t = np.cos(2 * theta)
        sin2t = np.sin(2 * theta)

        a = (cost / gammax) ** 2 + (sint / gammay) ** 2
        b = (sint / gammax) ** 2 + (cost / gammay) ** 2
        inv_gamma2 = 1 / gammax**2 - 1 / gammay**2
        c = 2 * sint * cost * inv_gamma2

        rad = a * diffx**2 + b * diffy**2 + c * diffx * diffy

        d_amp = (1 + rad) ** (-alpha)
        d_alpha = -amplitude * d_amp * np.log(1 + rad)

        f = -amplitude * alpha * (1 + rad) ** (-alpha - 1)
        d_x0 = f * (-2 * diffx * a - diffy * c)
        d_y0 = f * (-2 * diffy * b - diffx * c)
        d_theta = f * (
            diffx**2 * sin2t * inv_gamma2
            + 2 * diffx * diffy * inv_gamma2 * cos2t
            - diffy**2 * inv_gamma2 * sin2t
        )
        d_gammax = f * (
            -2
            / gammax**3
            * (cost**2 * diffx**2 + sint**2 * diffy**2 + diffx * diffy * sin2t)
        )
        d_gammay = f * (
            -2
            / gammay**3
            * (sint**2 * diffx**2 + cost**2 * diffy**2 - diffx * diffy * sin2t)
        )
        d_back = np.ones_like(d_x0)

        return [d_x0, d_y0, d_gammax, d_gammay, d_theta, d_alpha, d_amp, d_back]


def fit_moffat(data, err=None, guess=None, window=25, plot=False):
    if guess is None:
        cy, cx = np.unravel_index(np.nanargmax(data), data.shape)
        guess = (cx, cy)

    cutout = Cutout2D(data, guess, window)
    frame = cutout.data
    if err is not None:
        err_cutout = Cutout2D(err, guess, window)
        weights = 1 / err_cutout.data
    else:
        weights = None

    ys, xs = np.indices(frame.shape)
    fitter = modeling.fitting.LevMarLSQFitter(calc_uncertainties=True)
    inity, initx = np.unravel_index(frame.argmax(), frame.shape)
    model = Moffat(
        x0=initx,
        y0=inity,
        alpha=2,
        gammax=10,
        gammay=10,
        theta=0,
        amplitude=frame.max(),
        background=0,
    )
    model.x0.min = 0
    model.x0.max = window
    model.y0.min = 0
    model.y0.max = window
    model.alpha.min = 0.5
    model.alpha.max = 3
    model.gammax.min = 1
    model.gammax.max = window / 2
    model.gammay.min = 1
    model.gammay.max = window / 2
    model.theta.min = -np.pi / 4
    model.theta.max = np.pi / 4
    model.amplitude.min = 0
    model.amplitude.max = 2 * frame.max()
    model.background.fixed = True
    fit_model = fitter(
        model, xs, ys, frame, weights=weights, filter_non_finite=True, maxiter=5000
    )
    # plotting
    if plot:
        fig, axes = pro.subplots(ncols=3)
        axes[0].imshow(frame, cmap="bone", origin="lower")
        axes[1].imshow(
            fit_model(xs, ys),
            vmin=frame.min(),
            vmax=frame.max(),
            cmap="bone",
            origin="lower",
        )
        axes[2].imshow(
            frame - fit_model(xs, ys),
            vmin=-frame.max() / 10,
            vmax=frame.max() / 10,
            norm="div",
            origin="lower",
            cmap="div",
        )
        axes.format(xtickloc="none", ytickloc="none", grid=False)
        pro.show(block=True)
    # re-offset position
    ox, oy = cutout.origin_original
    fit_model.x0 += ox
    fit_model.y0 += oy
    return fit_model


def phot_from_model(data, err, model):
    if model.gammax > model.gammay:
        a = 2 * model.fwhmx
        b = 2 * model.fwhmy
        theta = model.theta.value
    else:
        a = 2 * model.fwhmy
        b = 2 * model.fwhmx
        theta = model.theta.value + np.pi / 2
    if theta > np.pi / 2:
        theta -= np.pi

    # plotting
    # fig, axes = pro.subplots(ncols=2)
    # cutout = Cutout2D(data, (model.x0.value, model.y0.value), 30, mode="partial")
    # bbox = cutout.bbox_original
    # axes[0].imshow(cutout.data, cmap="bone", origin="lower", extent=(*bbox[1], *bbox[0]))
    # ell = Ellipse((model.x0.value + 0.5, model.y0.value) + 0.5, 2 * a, 2 * b, angle=np.rad2deg(theta), edgecolor="r", fc="none")
    # axes[0].add_patch(ell)
    # ys, xs = np.mgrid[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    # axes[1].imshow(model(xs, ys), cmap="bone", origin="lower", extent=(*bbox[1], *bbox[0]))
    # ell = Ellipse((model.x0.value + 0.5, model.y0.value) + 0.5, 2 * a, 2 * b, angle=np.rad2deg(theta), edgecolor="r", fc="none")
    # axes[1].add_patch(ell)
    # axes.format(xtickloc="none", ytickloc="none", grid=False)
    # pro.show(block=True)

    flux = aperture_phot(
        data,
        err,
        model.x0.value,
        model.y0.value,
        a=a,
        b=b,
        theta=theta,
    )

    return flux
