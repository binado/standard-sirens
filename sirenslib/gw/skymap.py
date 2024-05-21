import numpy as np
import healpy as hp
from ligo.skymap.io.fits import read_sky_map
from pandas import read_csv

try:
    import pymaster as nmt
except ModuleNotFoundError:
    nmt = None


def get_skymaps(base_path, **kwargs):
    """Helper method to extract the skymaps published in the LVK GWTC-3 data release.
    Selects the skymaps tagged as `Default`.

    See https://zenodo.org/records/8177023

    Parameters
    ----------
    base_path : Path
        The path to the unzipped tar file downloaded from Zenodo

    Yields
    ------
    ndarray
        HEALPIX map describing the probability distribution of the sky localization of each event
    """
    csv_filename = list(base_path.glob("*.csv"))[0]
    event_table = read_csv(csv_filename)
    skymap_filenames = event_table.query("Default == True")["FileName"]
    for filename in skymap_filenames:
        filepath = base_path.joinpath(filename)
        maps, _ = read_sky_map(filepath, nest=False, **kwargs)
        yield maps


class GWSkymap:
    def __init__(self, probmap):
        """Create a new `GWSkymap` class instance.

        Parameters
        ----------
        probmap : ndarray
            HEALPIX map describing the probability distribution of the sky localization of the event
        """
        self.probmap = probmap
        self.npix = len(probmap)
        self.nside = hp.npix2nside(self.npix)

    def power_spectrum(self, lmax, method="anafast", nlb=4, remove_dipole=False, **kwargs):
        """Estimate the angular power spectrum from the probability map.

        Parameters
        ----------
        lmax: int
            The maximum multipole with which to compute the angular power spectrum
        method: {"anafast", "namaster"}, optional
            Method to compute the angular power spectrum, by default "anafast"
        nlb: int, optional
            Number of multipoles per bandpower setting in `namaster`. By default 4
        remove_dipole: bool, optional
            Remove the dipole before running `anafast`. By default False

        Returns
        -------
        ells: ndarray
            Array of multipoles
        cls: ndarray
            Array of $C_\ell$

        Raises
        ------
        ModuleNotFoundError
            Raised if pymaster package is not installed.
        """
        if method == "namaster":
            if nmt is None:
                raise ModuleNotFoundError("pymaster package is not installed.")

            # Create all-sky mask for the power spectrum
            mask = np.ones_like(self.probmap)
            field = nmt.NmtField(mask, [self.probmap])
            ellbins = nmt.NmtBin.from_nside_linear(self.nside, nlb)
            ells = ellbins.get_effective_ells()
            (cls,) = nmt.compute_full_master(field, field, ellbins, **kwargs)
        else:
            probmap = hp.remove_dipole(self.probmap) if remove_dipole else self.probmap
            cls = hp.anafast(self.probmap, use_pixel_weights=True, lmax=lmax, **kwargs)
            ells = np.arange(1, cls.size + 1)
        return ells, cls


def get_combined_skymap(gw_skymaps, nside):
    """Combine GW skymaps into a probability heatmap.

    Parameters
    ----------
    gw_skymaps : list[GWSkymap]
        List of skymaps, see `get_skymaps` method
    nside : int
        Target `nside` resolution parameter

    Returns
    -------
    GWSkymap
        The normalized probability heatmap
    """
    probmap = np.zeros(hp.nside2npix(nside))
    nmaps = len(gw_skymaps)
    for skymap in gw_skymaps:
        event_probmap = skymap.probmap if skymap.nside == nside else hp.ud_grade(skymap.probmap, nside, power=-2)
        probmap += event_probmap
    probmap /= nmaps
    assert np.allclose(np.sum(probmap), 1)
    return GWSkymap(probmap)
