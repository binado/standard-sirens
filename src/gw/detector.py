import os
from enum import Enum
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline

from .waveform import taylor_f2_waveform, taylor_f2_orientation
from .coordinates import Spherical3DCoordinates, Cartesian3DCoordinates
from .utils import snr, optimal_snr


dirname = os.getcwd()


class DetectorSensitivities(Enum):
    O5_DESIGN = "AplusDesign"
    LIGO_O4_HIGH = "aligo_O4high"
    LIGO_O4_LOW = "aligo_O4low"


class DetectorCoordinates(Enum):
    # Latitude, longitude, x-arm orientation, shape (in degrees)
    LIGO_HANFORD = (46.5, -119.4, -36, 90)
    LIGO_LIVINGSTON = (30.6, -90.8, -108, 90)
    VIRGO = (43.6, 10.5, 20, 90)
    KAGRA = (36.4, 137.3, 65, 90)


@dataclass
class DetectorPosition:
    """
    Class for storing a detector's position and orientation.
    """

    lat: float  # degrees
    lon: float  # degrees
    rot: float  # degrees
    shape: float  # degrees

    @property
    def theta(self):
        return np.pi / 2 - np.radians(self.lat)

    @property
    def phi(self):
        return np.radians(self.lon)

    @property
    def rot2rad(self):
        return np.radians(self.rot)

    def normal2detector(self) -> Spherical3DCoordinates:
        return Spherical3DCoordinates(self.theta, self.phi)

    def psi_detector_frame(self, alpha: float, beta: float, psi: float, iota: float) -> float:
        zvec = self.normal2detector().to_cartesian()
        nvec = Spherical3DCoordinates(alpha, beta).to_cartesian()  # Propagation direction
        # \hat{N} x \hat{L} lies on x-y plane in propagation frame and makes an angle psi with x
        nvec_cross_lvec = (
            Cartesian3DCoordinates(np.array([1, 0, 0])).rotate_vector(0, psi, 0).rotate_vector(alpha, beta, 0)
        )
        lvec = nvec_cross_lvec + nvec.scalarmul(np.cos(iota))
        # See Eq. 4.11 of de Souza, J. M. S. Late-time cosmology with third generation gravitational waves observatories.
        # (Rio Grande do Norte U., Universidade Federal do Rio Grande do Norte, Brasil, Rio Grande do Norte U., 2023).
        return np.arctan2(zvec * lvec - (nvec * lvec) * (zvec * nvec), zvec * nvec_cross_lvec)

    def pattern_function(self, alpha: float, beta: float, psi: float, iota: float):
        """
        Return the detector's pattern functions F+ and Fx given its position and orientation.
        Implemented for interferometers only.

        See Maggiore, M. Gravitational Waves. Vol. 1: Theory and Experiments.
        (Oxford University Press, 2007). doi:10.1093/acprof:oso/9780198570745.001.0001.
        Eqs. 7.31, 7.32
        Table 7.2

        Parameters
        ----------
        alpha : float
            pi / 2 - declination of propagation direction (radians)
        beta : float
            right ascension of propagation direction (radians)
        psi : float
            Polarization angle in geocentric frame
        iota : float
            Inclination angle of angular momentum vector with respect to direction of propagation

        Returns
        -------
        list
            F+ and Fx
        """
        psi_detframe = self.psi_detector_frame(alpha, beta, psi, iota)
        nvec = Spherical3DCoordinates(alpha, beta).to_cartesian()  # Propagation direction
        # Get alpha, beta in detector frame
        alphad, betad = Spherical3DCoordinates.from_cartesian(
            nvec.rotate_frame(self.theta, self.phi, self.rot2rad)
        ).angles
        # Compute F+ and F# for psi = 0
        cosalphad = np.cos(alphad)
        cos2betad = np.cos(2.0 * betad)
        sin2betad = np.sin(2.0 * betad)
        fplus0 = 0.5 * (1.0 + cosalphad**2) * cos2betad
        fcross0 = cosalphad * sin2betad
        cos2psi = np.cos(2.0 * psi_detframe)
        sin2psi = np.sin(2.0 * psi_detframe)
        # Psi rotation
        fplus = fplus0 * cos2psi - fcross0 * sin2psi
        fcross = fplus0 * sin2psi + fcross0 * cos2psi
        return fplus, fcross


def sensitivity_filepath(sensitivity):
    return os.path.join(dirname, f"../sensitivity/{sensitivity.value}.txt")


class Detector:
    def __init__(self, name: str, position: DetectorPosition, sensitivity: DetectorSensitivities) -> None:
        self.name = name
        self.position = position
        self.sensitivity_file = sensitivity_filepath(sensitivity)
        f_sn = np.loadtxt(self.sensitivity_file)
        self._f = f_sn[:, 0]
        self._sn = f_sn[:, 1] ** 2
        self.sn_interpolator = CubicSpline(self._f, self._sn)

    def snr(self, h, k, f):
        sn = self.sn_interpolator(f)
        return snr(h, k, sn, f)

    def optimal_snr(self, h, f):
        sn = self.sn_interpolator(f)
        return optimal_snr(h, sn, f)

    def taylor_f2_waveform(self, m1, m2, dl, alpha, beta, psi, iota, f):
        fplus, fcross = self.position.pattern_function(alpha, beta, psi, iota)
        return taylor_f2_waveform(m1, m2, dl, f, iota, fplus, fcross)

    def taylor_f2_waveform_fast(self, face_on_waveform, alpha, beta, psi, iota):
        fplus, fcross = self.position.pattern_function(alpha, beta, psi, iota)
        orientation = taylor_f2_orientation(iota, fplus, fcross)
        return face_on_waveform * orientation
