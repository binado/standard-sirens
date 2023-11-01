import numpy as np
from scipy.special import erf
from .utils import flat_cosmology, gaussian


class HierarchicalBayesianLikelihood:
    def __init__(self) -> None:
        pass


class SimplifiedLikelihood(HierarchicalBayesianLikelihood):
    """
    Simplified likelihood based on arxiv:2212.08694.
    """

    def __init__(self, sigma_constant, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma_constant = sigma_constant

    def redshift_likelihood(self, z, z_gal):
        """
        See Eq. (17) of arxiv:2212.08694
        """
        sigma = 0.0013 * (1 + z) ^ 3
        return gaussian(z, z_gal, sigma)

    def gw_likelihood(self, z, H0, dl):
        """
        See Eq. (21) of arxiv:2212.08694
        """
        sigma = self.sigma_constant * dl
        mean = flat_cosmology(H0).luminosity_distance(z)
        return gaussian(dl, mean, sigma)

    def detection_probability(self, z, H0, dl_th):
        """
        See Eq. (22) of arxiv:2212.08694
        """
        dl = flat_cosmology(H0).luminosity_distance(z)
        return 0.5 + 0.5 * erf((dl - dl_th) / np.sqrt(2) / self.sigma_constant / dl)
