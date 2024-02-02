import numpy as np
import h5py
import healpy as hp

DEFAULT_DATASETS = ["ra", "dec", "z", "mass", "skymap_indices"]


def traverse_file(group, node, target):
    if isinstance(node, dict):
        for key, obj in node.items():
            traverse_file(group[key], obj, target)
    elif isinstance(node, list):
        for dataset in node:
            setattr(target, dataset, group[dataset][()])
    elif isinstance(node, str):
        setattr(target, node, group[node][()])


def draw_galaxies(skymap, n_dir, alpha, n_min):
    """
    Return all galaxies within each randomly chosen n_dir directions.

    Each direction is guaranteed to have at least n_min galaxies
    """
    # Generate large enough sample of directions as some might be rejected
    n_sim = 10 * n_dir
    theta = np.random.uniform(0, np.pi / 2, n_sim)
    phi = np.random.uniform(0, 2 * np.pi, n_sim)

    current_dir = 0
    galaxies = []
    for i in range(n_sim):
        _, galaxies_at_direction = skymap.indices_at_direction(theta[i], phi[i], alpha)
        if np.sum(galaxies_at_direction) > n_min:
            current_dir += 1
            galaxies.append(galaxies_at_direction)
        if current_dir >= n_dir:
            return galaxies


class Skymap:
    def __init__(self, nside, ra, dec, **kwargs) -> None:
        self.ra = ra
        self.dec = dec
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self._indices = hp.ang2pix(nside, np.pi / 2.0 - dec, ra, **kwargs)

    def counts(self):
        """
        Return an array with the number of objects within each pixel
        """
        return np.bincount(self._indices, minlength=self.npix)

    def ang2pix(self, ra, dec, **kwargs):
        """
        Return the index of the skymap pixel that contains the
        coordinates (ra, dec)
        """
        return hp.ang2pix(self.nside, np.pi / 2.0 - dec, ra, **kwargs)

    def indices_at_direction(self, theta, phi, alpha):
        """
        Get all indices at an angular distance alpha from a sky direction
        (theta, phi).
        """
        center = hp.ang2vec(theta, phi)
        # Get corresponding HEALPIX pixels
        ipix_within_disc = hp.query_disc(nside=self.nside, vec=center, radius=alpha)
        indices_in_ipix_array = np.isin(self._indices, ipix_within_disc)
        return ipix_within_disc, indices_in_ipix_array


class GalaxyCatalog:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def get(self, dataset):
        with h5py.File(self.filename, "r") as f:
            return f[dataset][()]
