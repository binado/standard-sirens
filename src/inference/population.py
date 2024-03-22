import numpy as np

from src.utils.math import normalize as _normalize


def merger_rate_per_comoving_volume(z, alpha, beta, zp):
    """
    Return un-normalized Madau-Dickinson-like merger rate per comoving volume
    """
    return np.power(1 + z, alpha) / (1.0 + np.power((1 + z) / (1 + zp), alpha + beta))


def low_redshift_merger_rate_per_comoving_volume(z, alpha):
    """
    Return un-normalized power-law model for merger rate per comoving volume
    """
    return np.power(1 + z, alpha)


class UniformComovingRedshiftPrior:
    def __init__(self, cosmology, z) -> None:
        self.z = z
        # Pre-compute Jacobian (detector-frame, z) -> (source-frame, volume)
        self.dvc_dz_over_1pz = 4 * np.pi * cosmology.differential_comoving_volume(z).value / (1 + z)

    def __call__(self):
        return self.dvc_dz_over_1pz


class MadauDickinsonRedshiftPrior:
    def __init__(self, cosmology, z) -> None:
        self.z = z
        self.cosmology = cosmology
        # Pre-compute Jacobian (source-frame, z) -> (detector-frame, volume)
        self.dvc_dz_over_1pz = 4 * np.pi * cosmology.differential_comoving_volume(z).value / (1 + z)

    def __call__(self, alpha, beta, zp, normalize=False):
        """
        Return the merger rate prior on redshift, p(z|H0, alpha, beta, zp)

        Uses Madau-Dickinson-like binary formation rate
        See https://arxiv.org/abs/2003.12152
        """
        # H0 dependence will cancel out in normalization
        sfr = merger_rate_per_comoving_volume(self.z, alpha, beta, zp)
        # 1 + z factor in denominator accounts for the transformation from
        # detector frame time to source frame time
        p_cbc = self.dvc_dz_over_1pz * sfr
        # p_cbc is integrated over z in the likelihood numerator and denominator
        # Its overall normalization factor is cancelled out
        return _normalize(p_cbc, self.z) if normalize else p_cbc

    def eval(self, z, alpha, beta, zp, normalize=False):
        """
        Return the merger rate prior on redshift, p(z|H0, alpha, beta, zp) for arbitrary z array

        Not optimized for multiple evaluations
        """
        # H0 dependence will cancel out in normalization
        sfr = merger_rate_per_comoving_volume(z, alpha, beta, zp)
        # 1 + z factor in denominator accounts for the transformation from
        # detector frame time to source frame time
        dvc_dz_over_1pz = 4 * np.pi * self.cosmology.differential_comoving_volume(z).value / (1 + z)
        p_cbc = dvc_dz_over_1pz * sfr
        # p_cbc is integrated over z in the likelihood numerator and denominator
        # Its overall normalization factor is cancelled out
        return _normalize(p_cbc, z) if normalize else p_cbc
