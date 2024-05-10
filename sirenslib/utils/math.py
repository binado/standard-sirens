import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from .constants import eps


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
