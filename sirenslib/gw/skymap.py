import numpy as np
import healpy as hp
from ligo.skymap.io.fits import read_sky_map
from pandas import read_csv


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

    def power_spectrum(self, **kwargs):
        """Compute the angular power spectrum of the probability map with the `anafast` routine.
        Extra arguments are passed to `anafast`.

        Returns
        -------
        ndarray
            The array of $C_\ell$
        """
        return hp.anafast(self.probmap, use_pixel_weights=True, **kwargs)


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
