from abc import ABC, abstractmethod

import numpy as np
from astropy import constants as const
from astropy import units as u

from .utils import symmetric_mass_ratio, mchirp

TAYLOR_F2_IMPLEMENTED_PN_ORDER = 3.5
mpc_to_m = 1e6 * u.pc.to(u.m)
msun = const.M_sun.value
G = const.G.value
c = const.c.value


class FrequencyDomainWaveform(ABC):
    """
    Base class for frequency-domain waveform approximants.
    """

    def __init__(self, event):
        super().__init__()
        self._event = event
        self.f_isco = None

    @property
    def gw(self):
        return self._event

    @property
    def f_isco(self):
        mtot = msun * (self.gw.m1d + self.gw.m2d)
        risco = 6 * G * mtot / c**2
        return np.sqrt(G * mtot / risco**3) / np.pi

    @abstractmethod
    def polarizations(self, f):
        pass


class TaylorF2(FrequencyDomainWaveform):
    """Class implementing the TaylorF2 frequency-domain waveform approximant."""

    def __init__(self, event, pn_order=TAYLOR_F2_IMPLEMENTED_PN_ORDER):
        """Create a new TaylorF2 instance.

        Parameters
        ----------
        event : GWEvent
        pn_order : float, optional
            The order of the Post-Newtonian expansion, should be a half integer.
            By default TAYLOR_F2_IMPLEMENTED_PN_ORDER

        Raises
        ------
        NotImplementedError
            if pn_order is greater than TAYLOR_F2_IMPLEMENTED_PN_ORDER
        """
        super().__init__(event)
        if pn_order > TAYLOR_F2_IMPLEMENTED_PN_ORDER:
            raise NotImplementedError

        self.pn_order = pn_order

    def amplitude(self, f):
        gw = self.gw
        mchirp_in_kg = msun * mchirp(gw.m1d, gw.m2d)
        dl_in_m = gw.dl * mpc_to_m
        # Amplitude
        amp = (G * mchirp_in_kg) ** (5 / 6) * np.sqrt(5 / 24 / c**3) / dl_in_m / np.pi ** (2 / 3)
        return amp * f ** (-7 / 6)

    def phase_expansion(self, f):
        # Computing phase \Phi(f) up to 3.5 PN
        # Expansion on x = \pi G M f / c^3
        gw = self.gw
        eta = symmetric_mass_ratio(gw.m1d, gw.m2d)
        mtot = msun * (gw.m1d + gw.m2d)
        num_coeffs = int(2 * self.pn_order + 1)
        x = np.pi * G * mtot * f / c**3

        a = list(np.empty(8))
        a[0] = 1.0
        a[1] = 0.0
        a[2] = 3715 / 756 + 55 / 9 * eta
        a[3] = -16 * np.pi
        a[4] = 15293365 / 508032 + 27145 * eta / 504 + 3085 * eta**2 / 72
        a[5] = np.pi * (38645 / 756 - 65 * eta / 9) * (1.0 + np.log(6**1.5 * x))
        a[6] = (
            11583231236531 / 4694215680
            - (640 / 3) * np.pi**2
            - 6848 * np.euler_gamma / 21
            + (-15737765635 / 3048192 + 2255 * np.pi**2 / 12) * eta
            + 76055 * eta**2 / 1728
            - 127825 * eta**3 / 1296
            - 6848 / 63 * np.log(64 * x)
        )
        a[7] = np.pi * (77096675 / 254016 + 378515 * eta / 1512 - 74045 * eta**2 / 756)

        phase = 0.0
        for i in range(num_coeffs):
            phase += a[i] * np.power(x, i / 3)
        phase *= 3 / 128 / eta / np.power(x, 5 / 3)
        return phase

    def polarizations(self, f):
        """
        Compute frequency-domain TaylorF2 waveform approximant up to 3.5 PN

        See Eq. (129) of Sathyaprakash, B. S. & Schutz, B. F.
        Physics, Astrophysics and Cosmology with Gravitational Waves.
        Living Rev. Relativ. 12, 2 (2009).
        """
        gw = self.gw
        plus_phase = self.phase_expansion(f) - np.pi / 4
        cross_phase = plus_phase + np.pi / 2
        amp = self.amplitude(f)

        # Cutoff
        cutoff = np.ones_like(f) * (f < 4 * self.f_isco)

        # Polarization due to orientation
        hplus, hcross = 0.5 * (1.0 + np.cos(gw.iota) ** 2), np.cos(gw.iota)
        hplus *= cutoff * amp * np.exp(plus_phase * 1.0j)
        hcross *= cutoff * amp * np.exp(cross_phase * 1.0j)

        return hplus, hcross
