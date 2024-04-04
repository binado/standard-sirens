import numpy as np
from astropy import constants as const
from astropy import units as u

from .utils import symmetric_mass_ratio, mchirp

mpc_to_m = 1e6 * u.pc.to(u.m)
msun = const.M_sun.value


def taylor_f2_waveform_face_on(m1, m2, dl, f, pn_order=3.5):
    """
    Compute frequency-domain TaylorF2 waveform approximant up to 3.5 PN

    See Eq. (129) of Sathyaprakash, B. S. & Schutz, B. F.
    Physics, Astrophysics and Cosmology with Gravitational Waves.
    Living Rev. Relativ. 12, 2 (2009).
    """
    eta = symmetric_mass_ratio(m1, m2)
    mtot = msun * (m1 + m2)
    mchirp_in_kg = msun * mchirp(m1, m2)
    num_coeffs = int(2 * pn_order + 1)
    dl_in_m = dl * mpc_to_m

    # Computing phase \Phi(f) up to 3.5 PN
    # Expansion on x = \pi G M f / c^3
    G = const.G.value
    c = const.c.value
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
    phase -= np.pi / 4

    # Amplitude
    amp = (G * mchirp_in_kg) ** (5 / 6) * np.sqrt(5 / 24 / c**3) / dl_in_m / np.pi ** (2 / 3)

    # Cutoff
    risco = 6 * G * mtot / c**2
    fisco = np.sqrt(G * mtot / risco**3) / np.pi
    cutoff = np.ones_like(f) * (f < 4 * fisco)

    return cutoff * amp * f ** (-7 / 6) * np.exp(phase * 1.0j), fisco


def taylor_f2_orientation(iota, fplus, fcross):
    # Orientation for [plus, cross] polarizations
    hplus, hcross = 0.5 * (1.0 + np.cos(iota) ** 2), np.cos(iota)
    # Amplitude gets factor (F+ h+, Fx hx)^2
    # pattern_function = np.array([fplus, fcross])
    # orientation_factor = pattern_function * orientation
    # return np.sqrt(np.dot(orientation_factor, orientation_factor))
    return hplus * fplus + hcross * fcross


def taylor_f2_waveform(m1, m2, dl, f, iota, fplus, fcross):
    waveform_face_on, fisco = taylor_f2_waveform_face_on(m1, m2, dl, f)
    orientation = taylor_f2_orientation(iota, fplus, fcross)
    return orientation * waveform_face_on, fisco
