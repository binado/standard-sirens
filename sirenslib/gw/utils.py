import numpy as np
from scipy.integrate import simpson


def symmetric_mass_ratio(m1, m2):
    return m1 * m2 / (m1 + m2) ** 2


def mchirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def mchirp_eta2masses(m, eta):
    total_mass = np.power(eta, -0.6) * m
    sqrt_quadratic_delta = np.sqrt(total_mass**2 - 4 * eta * total_mass)
    m_1 = 0.5 * (total_mass + sqrt_quadratic_delta)
    m_2 = 0.5 * (total_mass - sqrt_quadratic_delta)
    return m_1, m_2


def scalar_product(a, b, sn, f):
    """
    Return the signal-scalar product in frequency space.

    Parameters
    ----------
    a : array_like
        First signal in frequency domain
    b : array_like
        Second signal in frequency domain
    sn : array_like
        Noise spectral density
    f : array_like
        Frequency array over which integration will be carried over.
        f should be in [f_min, f_max] range of the sn.

    Returns
    -------
    float
        Signal-scalar product of a and b
    """
    return 4 * np.real(simpson(a * np.conj(b) / sn, x=f))


def snr(h, k, sn, f):
    """
    Return signal-to-noise ratio for signal h and filter function k.

    Parameters
    ----------
    h : array_like
        Signal in frequency domain
    k : array_like
        Filter function in frequency domain
    sn : array_like
        Noise spectral density
    f : array_like
        Frequency array over which integration will be carried over.
        f should be in [f_min, f_max] range of the sn.

    Returns
    -------
    float
        Signal-to-noise ratio between h and k
    """
    u = 0.5 * sn * k
    s = scalar_product(h, u, sn, f)
    n = np.sqrt(scalar_product(u, u, sn, f))
    return s / n


def optimal_snr(h, sn, f):
    """
    Return optimal signal-to-noise ratio for a signal h.

    Parameters
    ----------
    h : array_like
        Signal in frequency domain
    sn : array_like
        Noise spectral density
    f : array_like
        Frequency array over which integration will be carried over.
        f should be in [f_min, f_max] range of the sn.

    Returns
    -------
    float
        Optimal signal-to-noise ration for h
    """
    return np.sqrt(scalar_product(h, h, sn, f))


def network_snr(*snrs):
    return np.sqrt(np.sum([snr**2 for snr in snrs]))
