from dataclasses import dataclass

import numpy as np

from .coordinates import Cartesian3DCoordinates, Spherical3DCoordinates
from ..utils.cosmology import luminosity_distance

default_spin_vector = np.array([0, 0, 0])


@dataclass
class GWEvent:
    z: float
    dl: float
    m1s: float
    m2s: float
    position: Spherical3DCoordinates  # In geocentric frame
    psi: float  # In geocentric frame
    iota: float
    t_coal: float = 0.0
    phi_coal: float = 0.0
    s1: Cartesian3DCoordinates = Cartesian3DCoordinates(default_spin_vector)
    s2: Cartesian3DCoordinates = Cartesian3DCoordinates(default_spin_vector)

    @property
    def m1d(self):
        return self.m1s * (1 + self.z)

    @property
    def m2d(self):
        return self.m2s * (1 + self.z)

    def z_to_dl(self, cosmology, z, **kwargs):
        self.dl = luminosity_distance(cosmology, z, **kwargs)
        return self.dl

    def to_gwdali_dict(self):
        theta, phi = self.position.angles
        return {
            "z": self.z,
            "DL": self.dl * 1e-3,  # Gpc
            "m1": self.m1d,
            "m2": self.m2d,
            "RA": np.degrees(phi),
            "Dec": np.degrees(np.pi / 2 - theta),
            "iota": self.iota,
            "psi": self.psi,
            "t_coal": self.t_coal,
            "phi_coal": self.phi_coal,
            "sx1": self.s1.x,
            "sy1": self.s1.y,
            "sz1": self.s1.z,
            "sx2": self.s2.x,
            "sy2": self.s2.y,
            "sz2": self.s2.z,
        }


class GWEventGenerator:
    """TODO: class methods not yet updated to work with `gw.GWEvent`."""

    def __init__(self, z_draw_max=1.4, dl_th=1550) -> None:
        self.z_draw_max = z_draw_max
        self.dl_th = dl_th
        self.drawn_redshifts = None

    def uniform_p_rate(self, z):
        p_rate = np.zeros_like(z)
        p_rate[z <= self.z_draw_max] = 1
        return p_rate / np.sum(p_rate)

    def weighted_p_rate(self, z, weights):
        p_rate = np.copy(weights)
        p_rate[z > self.z_draw_max] = 0
        return p_rate / np.sum(p_rate)

    def p_rate(self, z_gal, weights=None):
        return self.weighted_p_rate(z_gal, weights) if weights is not None else self.uniform_p_rate(z_gal)

    def draw_redshifts(self, z_gal, n):
        p_rate = self.uniform_p_rate(z_gal)
        self.drawn_redshifts = np.random.choice(z_gal, n, p=p_rate)

    def from_redshifts(self, cosmology, z, sigma_dl, **kwargs):
        n = len(z)

        # Convert them into "true" gw luminosity distances using a fiducial cosmology
        true_dl = luminosity_distance(cosmology, z)

        # Convert true gw luminosity distances into measured values
        # drawn from a normal distribution consistent with the GW likelihood
        sigma = true_dl * sigma_dl
        observed_dl = true_dl + sigma * np.random.standard_normal(n)
        # Filter events whose dL exceeds threshold
        detectable_dl = observed_dl[observed_dl < self.dl_th]
        return [GWEvent(dl=dl, **kwargs) for dl in detectable_dl]

    def from_catalog(self, cosmology, los, sigma_dl, n_gw, weights=None):
        # z_gal is an array of galaxy redshifts in one sky direction
        # Merger probability on 0 < z < z_draw_max
        # Weighted by galaxy mass if weights is provided
        # Uniform otherwise
        z_gal = los.z
        p_rate = self.p_rate(z_gal, weights)

        # Get the "true" gw redshifts
        drawn_gw_zs = np.random.choice(z_gal, n_gw, p=p_rate)

        # Get the measured redshifts
        return self.from_redshifts(cosmology, drawn_gw_zs, sigma_dl, los=los)
