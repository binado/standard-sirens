import os
from enum import Enum
from dataclasses import dataclass
import numpy as np

import src.gw.utils as utils


dirname = os.getcwd()


class DetectorSensitivities(Enum):
    O5_DESIGN = "AplusDesign"
    LIGO_O4_HIGH = "aligo_O4high"
    LIGO_O4_LOW = "aligo_O4low"


class DetectorCoordinates(Enum):
    LIGO_HANFORD = (46.5, -119.4, -36)
    LIGO_LIVINGSTON = (30.6, -90.8, -108)
    VIRGO = (43.6, 10.5, 20)
    KAGRA = (36.4, 137.3, 65)


@dataclass
class DetectorPosition:
    """
    Class for storing a detector's position and orientation.
    """

    lat: float  # degrees
    lon: float  # degrees
    orientation: float  # degrees

    @property
    def theta(self):
        return np.pi / 2 - np.radians(self.lat)

    @property
    def phi(self):
        return np.radians(self.lon)

    @property
    def psi(self):
        return np.radians(self.orientation)

    def _aligned_pattern_function(self):
        costheta = np.cos(self.theta)
        cos2phi = np.cos(2.0 * self.phi)
        sin2phi = np.sin(2.0 * self.phi)
        fplus = 0.5 * (1.0 + costheta**2) * cos2phi
        fcross = costheta * sin2phi
        return fplus, fcross

    def pattern_function(self):
        """
        Return the detector's pattern functions F+ and Fx given its position and orientation.
        Implemented for interferometers only.

        See Maggiore, M. Gravitational Waves. Vol. 1: Theory and Experiments.
        (Oxford University Press, 2007). doi:10.1093/acprof:oso/9780198570745.001.0001.
        Eqs. 7.31, 7.32
        Table 7.2

        Returns
        -------
        list
            F+ and Fx
        """
        cos2psi = np.cos(2.0 * self.psi)
        sin2psi = np.sin(2.0 * self.psi)
        fplus0, fcross0 = self._aligned_pattern_function()
        fplus = fplus0 * cos2psi - fcross0 * sin2psi
        fcross = fplus0 * sin2psi + fcross0 * cos2psi
        return fplus, fcross


def sensitivity_filepath(sensitivity):
    return os.path.join(dirname, f"../../{sensitivity.value}.txt")


class Detector:
    def __init__(self, name: str, position: DetectorPosition, sensitivity: DetectorSensitivities) -> None:
        self.name = name
        self.position = position
        self.sensitivity_file = sensitivity_filepath(sensitivity)
        self.f, self.sn = np.loadtxt(self.sensitivity_file)

    def snr(self, h, k):
        return utils.snr(h, k, self.sn, self.f)

    def optimal_snr(self, h):
        return utils.optimal_snr(h, self.sn, self.f)

    def taylor_f2_waveform(self, m1, m2, dl, iota):
        fplus, fcross = self.position.pattern_function()
        return utils.taylor_f2_waveform(m1, m2, dl, iota, fplus, fcross, self.f)
