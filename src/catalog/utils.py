import numpy as np
import h5py
import healpy as hp

DEFAULT_DATASETS = ["ra", "dec", "z", "mass", "skymap_indices"]


class GalaxyCatalog:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def get(self, dataset):
        with h5py.File(self.filename, "r") as f:
            return f[dataset][()]


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

    def indices_at_pixels(self, pixels):
        """
        Return indices of galaxies located within given pixels
        """
        return np.isin(self._indices, pixels)

    def average_at_pixel(self, array, pixel):
        indices_at_pixel = self.indices_at_pixels([pixel])
        return np.average(array[indices_at_pixel])

    def pixels_at_direction(self, theta, phi, alpha):
        """
        Get all indices at an angular distance alpha from a sky direction
        (theta, phi).
        """
        center = hp.ang2vec(theta, phi)
        # Get corresponding HEALPIX pixels
        return hp.query_disc(nside=self.nside, vec=center, radius=alpha)

    @staticmethod
    def bin_array(array, bins, **kwargs):
        indices = np.digitize(array, bins, **kwargs)
        nbins = len(bins)
        masks = [indices == i + 1 for i in range(nbins)]
        return masks

    @staticmethod
    def remove_mask(array, mask):
        return np.ma.array(array, mask=mask).compressed()


class MaskedMap:
    """
    Class that represents a masked map for healpy visualization.
    """

    def __init__(self, full_map, mask=None, q=None, map_dtype=np.float64) -> None:
        # Float conversion necessary for accomodating mask value
        self.full_map = full_map.astype(map_dtype)
        self.mask = self.compute_mask(mask, q)
        # Filling these pixels with the special value for a mask in healpy
        self.masked_map = np.ma.MaskedArray(self.full_map, ~self.mask, fill_value=hp.UNSEEN)
        self.fsky = len(self.masked_map.compressed()) / len(self.full_map)

    def compute_mask(self, mask=None, q=None):
        if mask is not None:
            return mask
        if q is not None:
            return self.full_map >= np.quantile(self.full_map, q)
        raise ValueError


class LineOfSight:
    """
    Class that represents a group of galaxies in a given line of sight.
    """

    def __init__(self, skymap, pixels) -> None:
        # Create mask with objects in given pixels
        self.indices = skymap.indices_at_pixels(pixels)
        self.ngalaxies = np.sum(self.indices)
        self.default_datasets = dict(z="z_cmb")
        self.z = None
        self.weights = None
        self.p_gal = None

    def fetch(self, catalog, strip_null=True, weights=None, **datasets):
        """
        Fetch catalog datasets for the given line of sight.
        """
        datasets.update(**self.default_datasets)
        # Get full all-sky datasets
        columns_at_los = {column: catalog.get(dataset)[self.indices] for column, dataset in datasets.items()}
        # Build mask for LOS
        mask = np.logical_and.reduce(np.array([np.isfinite(column) for column in columns_at_los.values()]))
        if strip_null:
            for column in columns_at_los:
                columns_at_los[column] = columns_at_los[column][mask]
        for column, dataset in columns_at_los.items():
            setattr(self, column, dataset)

        if weights is not None:
            self.weights = columns_at_los[weights]


def draw_galaxies(skymap: Skymap, n_dir: int, alpha, n_min: int):
    """
    Return all galaxies within each randomly chosen n_dir directions.

    Each direction is guaranteed to have at least n_min galaxies
    """
    # Generate large enough sample of directions as some might be rejected
    n_sim = 10 * n_dir
    theta = np.random.uniform(0, np.pi / 2, n_sim)
    phi = np.random.uniform(0, 2 * np.pi, n_sim)

    current_dir = 0
    lines_of_sight = []
    for i in range(n_sim):
        pixels = skymap.pixels_at_direction(theta[i], phi[i], alpha)
        los = LineOfSight(skymap, pixels)
        if los.ngalaxies > n_min:
            lines_of_sight.append(los)
            current_dir += 1

        if current_dir >= n_dir:
            break

    return lines_of_sight
