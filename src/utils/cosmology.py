from astropy.cosmology import FlatLambdaCDM
from .constants import Om0


def flat_cosmology(H0):
    return FlatLambdaCDM(H0, Om0)


def luminosity_distance(cosmology, z):
    return cosmology.luminosity_distance(z).to("Mpc").value
