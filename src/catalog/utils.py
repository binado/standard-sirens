import numpy as np
import h5py
import healpy as hp

DEFAULT_DATASETS = ["ra", "dec", "z", "mass", "skymap_indices"]


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
    def __init__(self, filename) -> None:
        self.filename = filename
        self.file = h5py.File(filename, "r")

    def dataset(self, group: str, name: str):
        path = "{0}/{1}".format(group, name)
        # The '()' slice retrieves all data from the dataset
        return self.file[path][()]

    def close(self):
        self.file.close()

    def npix(self, nside):
        return hp.nside2npix(nside)

    def get_skymap(self, nside, ra, dec, **kwargs):
        return hp.ang2pix(nside, dec + np.pi / 2, ra, **kwargs)

    def galaxy_counts(self, skymap_indices):
        """
        Return an array with the number of galaxies within each pixel
        """
        nonempty_pixels, counts = np.unique(skymap_indices, return_counts=True)
        skymap = np.zeros(self.npix)
        for pixel, count in zip(nonempty_pixels, counts):
            skymap[pixel] = count

        return skymap

    def galaxies_at_direction(self, skymap_indices, theta, phi, alpha):
        """
        Get all galaxy redshifts at an angular distance alpha from a sky direction
        (theta, phi).
        """
        center = hp.ang2vec(theta, phi)
        # Get corresponding HEALPIX pixels
        ipix_within_disc = hp.query_disc(nside=self.nside, vec=center, radius=alpha)
        galaxies_in_ipix_array = np.isin(skymap_indices, ipix_within_disc)
        return ipix_within_disc, galaxies_in_ipix_array

    def draw_galaxies(self, skymap_indices, n_dir, alpha, n_min):
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
            _, galaxies_at_direction = self.galaxies_at_direction(skymap_indices, theta[i], phi[i], alpha)
            if np.sum(galaxies_at_direction) > n_min:
                current_dir += 1
                galaxies.append(galaxies_at_direction)
            if current_dir >= n_dir:
                return galaxies
