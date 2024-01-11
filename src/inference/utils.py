import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from astropy.cosmology import FlatLambdaCDM
from .constants import Om0, eps


def normalize(y, x, safe=False):
    """Return an array normalized with its integral.

    Parameters
    ----------
    y : array_like
        The array to normalize
    x : array_like
        The axis to perform the integration
    safe : bool, optional
        Whether to prevent division by zero-norm, by default False

    Returns
    -------
    array_like
        The normalized array
    """
    norm = simpson(y, x)
    if safe:
        norm += eps
    return y / norm


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) ** 2) / sigma**2) / np.sqrt(2 * np.pi) / sigma


def lognormal(x, mu, sigma):
    return gaussian(np.log(x), mu, sigma) / x


def flat_cosmology(H0):
    return FlatLambdaCDM(H0, Om0)


def merger_rate(z, alpha, beta, c):
    return np.power(1 + z, alpha) / (1.0 + np.power((1 + z) / c, beta))


def sample_from_func(n, func, x, *args, **kwargs):
    """Return samples from a function with known PDF using the inversion technique.

    Parameters
    ----------
    n : int
        The number of samples to return
    func : function
        The distribution PDF. Does not need to be normalized
    x : array_like
        The array from which the samples will be drawn

    Returns
    -------
    array_like
        The array of samples with length n
    """
    prob = func(x, *args, **kwargs)
    cdf = cumulative_trapezoid(prob, x)
    max_cdf = cdf[-1]
    percentiles = np.random.uniform(high=max_cdf, size=n)
    return x[np.searchsorted(cdf, percentiles)]


class EventGenerator:
    def __init__(self, H0, z_draw_max, dl_th) -> None:
        self.cosmology = flat_cosmology(H0)
        self.z_draw_max = z_draw_max
        self.dl_th = dl_th
        self.drawn_redshifts = None

    def luminosity_distance(self, cosmology, z):
        return cosmology.luminosity_distance(z).to("Mpc").value

    def uniform_p_rate(self, z_gal, z_draw_max):
        # Uniform merger probability on 0 < z < z_draw_max
        p_rate = np.zeros_like(z_gal)
        p_rate[z_gal <= z_draw_max] = 1
        p_rate /= np.sum(p_rate)
        return p_rate

    def draw_redshifts(self, z_gal, n):
        p_rate = self.uniform_p_rate(z_gal, self.z_draw_max)
        self.drawn_redshifts = np.random.choice(z_gal, n, p=p_rate)

    def from_catalog(self, cosmology, sigma_dl, noise=None):
        drawn_gw_zs = self.drawn_redshifts
        if drawn_gw_zs is None:
            raise ValueError("drawn_redshifts method must be called before.")

        # Convert them into "true" gw luminosity distances using a fiducial cosmology
        drawn_gw_dls = self.luminosity_distance(cosmology, drawn_gw_zs)
        _noise = noise if noise is not None else np.random.standard_normal(len(drawn_gw_dls))

        # Convert true gw luminosity distances into measured values
        # drawn from a normal distribution consistent with the GW likelihood
        sigma = drawn_gw_dls * sigma_dl
        observed_gw_dls = _noise * sigma + drawn_gw_dls
        # Filter events whose dL exceeds threshold
        return observed_gw_dls[observed_gw_dls < self.dl_th]
