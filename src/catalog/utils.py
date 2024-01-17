import numpy as np
import h5py
import healpy as hp

DEFAULT_DATASETS = ["ra", "dec", "z", "mass", "skymap_indices"]


class GalaxyCatalog:
    def __init__(self, filename, skymap_indices_dataset="skymap") -> None:
        self.filename = filename

        self.file = h5py.File(filename, "r")
        self.nside = self.file.attrs["nside"]
        self.npix = hp.nside2npix(self.nside)
        # self.nest_from_dataset = self.file.attrs["nest"]

        if skymap_indices_dataset not in self.file.keys():
            raise ValueError(f"Skymap indices dataset named '{skymap_indices_dataset}' does not exist in file.")
        self.skymap_indices_dataset = skymap_indices_dataset
        for dataset_name, data in self.file.items():
            setattr(self, dataset_name, data[...])

    @property
    def skymap_indices(self):
        return getattr(self, self.skymap_indices_dataset)

    def close(self):
        self.file.close()

    def galaxy_counts(self):
        """
        Return an array with the number of galaxies within each pixel
        """
        nonempty_pixels, counts = np.unique(self.skymap_indices, return_counts=True)
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

    def galaxies_at_direction(self, theta, phi, alpha):
        """
        Get all galaxy redshifts at an angular distance alpha from a sky direction
        (theta, phi).
        """
        center = hp.ang2vec(theta, phi)
        # Get corresponding HEALPIX pixels
        ipix_within_disc = hp.query_disc(nside=self.nside, vec=center, radius=alpha)
        galaxies_in_ipix_array = np.isin(self.skymap_indices, ipix_within_disc)
        return ipix_within_disc, galaxies_in_ipix_array

    def draw_galaxies(self, n_dir, alpha, n_min):
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
            _, galaxies_at_direction = self.galaxies_at_direction(theta[i], phi[i], alpha)
            if np.sum(galaxies_at_direction) > n_min:
                current_dir += 1
                galaxies.append(galaxies_at_direction)
            if current_dir >= n_dir:
                return galaxies
