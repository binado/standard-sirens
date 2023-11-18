import numpy as np
from astropy.cosmology import FlatLambdaCDM
from .constants import Om0


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) ** 2) / sigma**2) / np.sqrt(2 * np.pi) / sigma


def lognormal(x, mu, sigma):
    return gaussian(np.log(x), mu, sigma) / x


def flat_cosmology(H0):
    return FlatLambdaCDM(H0, Om0)
