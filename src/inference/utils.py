import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from astropy.cosmology import FlatLambdaCDM
from .constants import Om0, eps


def normalize(y, x, safe=True, min_norm=eps):
    """Return an array normalized with its integral.

    Parameters
    ----------
    y : array_like
        The array to normalize
    x : array_like
        The axis to perform the integration
    safe : bool, optional
        Whether to prevent division by zero-norm, by default True
    min_norm : float, optional
        Minimum norm in case of save division, by default 1e-10

    Returns
    -------
    array_like
        The normalized array

    Raises
    ------
    ZeroDivisionEror
    """
    norm = simpson(y, x)
    try:
        res = y / norm
        return res
    except ZeroDivisionError as err:
        if safe:
            return y / (norm + min_norm)
        else:
            raise err


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) ** 2) / sigma**2) / np.sqrt(2 * np.pi) / sigma


def lognormal(x, mu, sigma):
    return gaussian(np.log(x), mu, sigma) / x


def flat_cosmology(H0):
    return FlatLambdaCDM(H0, Om0)


def luminosity_distance(cosmology, z):
    return cosmology.luminosity_distance(z).to("Mpc").value


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
    def __init__(self, z_draw_max=1.4, dl_th=1550) -> None:
        self.z_draw_max = z_draw_max
        self.dl_th = dl_th
        self.drawn_redshifts = None

    def uniform_p_rate(self, z):
        p_rate = np.zeros_like(z)
        p_rate[z <= self.z_draw_max] = 1
        return p_rate / np.sum(p_rate)

    def mass_weighted_p_rate(self, z, weights):
        p_rate = np.copy(weights)
        p_rate[z > self.z_draw_max] = 0
        return p_rate / np.sum(p_rate)

    def p_rate(self, z_gal, weights=None):
        return self.mass_weighted_p_rate(z_gal, weights) if weights is not None else self.uniform_p_rate(z_gal)

    def draw_redshifts(self, z_gal, n):
        p_rate = self.uniform_p_rate(z_gal)
        self.drawn_redshifts = np.random.choice(z_gal, n, p=p_rate)

    def from_redshifts(self, cosmology, z, sigma_dl):
        n = len(z)

        # Convert them into "true" gw luminosity distances using a fiducial cosmology
        true_dl = luminosity_distance(cosmology, z)

        # Convert true gw luminosity distances into measured values
        # drawn from a normal distribution consistent with the GW likelihood
        sigma = true_dl * sigma_dl
        observed_dl = true_dl + sigma * np.random.standard_normal(n)
        # Filter events whose dL exceeds threshold
        return observed_dl[observed_dl < self.dl_th]

    def from_catalog(self, cosmology, z_gal, sigma_dl, n_gw: int, weights=None):
        if weights is not None:
            assert len(z_gal) == len(weights), "z_gal and weights should have the same shape"

        try:
            _ = len(z_gal[0])
            # z_gal is a list of arrays of galaxy redshifts in different sky direction
            events = []
            for i, z_i in enumerate(z_gal):
                mass_i = weights[i] if weights is not None else None
                p_rate = self.p_rate(z_i, mass_i)
                drawn_gw_zs = np.random.choice(z_i, n_gw, p=p_rate)
                events.append(self.from_redshifts(cosmology, drawn_gw_zs, sigma_dl))
        except TypeError:
            # z_gal is an array of galaxy redshifts in one sky direction
            # Merger probability on 0 < z < z_draw_max
            # Weighted by galaxy mass if weights is provided
            # Uniform otherwise
            p_rate = self.p_rate(z_gal, weights)

            # Get the "true" gw redshifts
            drawn_gw_zs = np.random.choice(z_gal, n_gw, p=p_rate)

            # Get the measured redshifts
            events = self.from_redshifts(cosmology, drawn_gw_zs, sigma_dl)

        return events
