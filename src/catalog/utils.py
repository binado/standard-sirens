import numpy as np
import h5py
import healpy as hp

DEFAULT_DATASETS = ["ra", "dec", "z", "skymap_indices"]


class GalaxyCatalog:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.file = h5py.File(filename, "r")
        self.nside = self.file.attrs["nside"]
        self.npix = hp.nside2npix(self.nside)
        # self.nest_from_dataset = self.file.attrs["nest"]
        for dataset_name, data in self.file.items():
            setattr(self, dataset_name, data[...])

    def close(self):
        self.file.close()

    def galaxy_counts(self, skymap_indices_dataset="skymap"):
        """
        Return an array with the number of galaxies within each pixel
        """
        skymap_indices = getattr(self, skymap_indices_dataset)
        nonempty_pixels, counts = np.unique(skymap_indices, return_counts=True)
        skymap = np.zeros(self.npix)
        for pixel, count in zip(nonempty_pixels, counts):
            skymap[pixel] = count

        return skymap

    def indices(self, ra, dec, nest):
        """
        Return the index of the skymap pixel that contains the
        coordinates (ra, dec)
        """
        return hp.ang2pix(self.nside, np.pi / 2.0 - dec, ra, nest=nest)

    def z_at_index(self, ipix_array, skymap_indices_dataset="skymap"):
        """
        Return an array of redshifts of all galaxies within pixel of index ipix
        """
        skymap_indices = getattr(self, skymap_indices_dataset)
        z = getattr(self, "z")
        galaxies_in_ipix_array = np.isin(skymap_indices, ipix_array)
        return z[galaxies_in_ipix_array]

    def get_redshifts_at_direction(self, theta, phi, alpha):
        """
        Get all galaxy redshifts at an angular distance alpha from a sky direction
        (theta, phi).
        """
        center = hp.ang2vec(theta, phi)
        # Get corresponding HEALPIX pixels
        ipix_within_disc = hp.query_disc(nside=self.nside, vec=center, radius=alpha)
        # Get corresponding galaxy redshifts
        return self.z_at_index(ipix_within_disc)

    def draw_redshifts(self, n_dir, alpha, n_min):
        """
        Return a generator with a list of galaxy redshifts for n_dir directions

        Each direction is guaranteed to have at least n_min galaxies
        """
        theta = np.random.uniform(0, np.pi / 2, 10 * n_dir)
        phi = np.random.uniform(0, 2 * np.pi, 10 * n_dir)
        current_dir = 0
        while current_dir < n_dir:
            redshifts = self.get_redshifts_at_direction(
                theta[current_dir], phi[current_dir], alpha
            )
            if len(redshifts) > n_min:
                current_dir += 1
                yield redshifts
