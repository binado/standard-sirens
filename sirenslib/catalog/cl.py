import numpy as np
import pymaster as nmt


class AngularPowerSpectrumEstimator:
    def __init__(self, full_skymap, bin_skymaps, include_full_skymap=False):
        """Create a new class instance from an existing skymap and a corresponding skymap for each bin.

        Parameters
        ----------
        full_skymap : catalog.utils.Skymap
            Skymap representing the full (unbinned) catalog
        bin_skymaps : list[catalog.utils.Skymap]
            A list of Skymap instances for each bin
        include_full_skymap : bool, optional
            Whether to include the full skymap in the list of maps, by default False
        """
        self.full_skymap = full_skymap
        self.full_map = full_skymap.counts()
        self.bin_skymaps = bin_skymaps
        self.maps = [skymap.counts() for skymap in bin_skymaps]
        if include_full_skymap:
            self.maps.append(full_skymap.counts())

    def mask(self, quantile_cut, aposize=None, **kwargs):
        """Return a mask of the full catalog by discarding the least populated pixels.
        Extra arguments are passed to the `pymaster.mask_apodization` method.

        Parameters
        ----------
        quantile_cut : float
            The pixels lying below this quantile will be discarded by the mask.
        aposize : float, optional
            The `aposize` parameter in the `pymaster.mask_apodization` method

        Returns
        -------
        mask: array_like
            The mask returned by `pymaster.mask_apodization`

        Raises
        ------
        ValueError
            Raised if quatile_cut is not betwen 0 and 1
        """
        if quantile_cut < 0 or quantile_cut > 1:
            raise ValueError

        mask = self.full_map >= np.quantile(self.full_map, quantile_cut)
        if aposize is not None:
            mask = nmt.mask_apodization(mask, aposize=aposize, **kwargs)
        return mask

    def fields(self, mask):
        """Return a list of instances of `pymaster.NmtField` for the maps corresponding to
        the number counts of each bin that was passed to the class instance.

        Parameters
        ----------
        mask : array_like
            The mask for the corresponding fields. See `mask` method

        Returns
        -------
        list[pymaster.NmtField]
            List of fields created from the mask and each map
        """
        return [nmt.NmtField(mask, [mapi]) for mapi in self.maps]

    def auto_cls(self, l_per_bandpower, fields):
        """Generate auto power spectra for different fields.

        Parameters
        ----------
        l_per_bandpower : int
            Number of multipoles per bin
        fields : list[nmt.NmtField]
            the list of fields with which to calculate the C_\ell. See `fields` method

        Returns
        -------
        ell: array_like
            Array of effective multipoles
        cls: array_like
            Array of calculated C_\ell. If ell has shape (N,), this has shape (nfields, N)
        """
        ellbins = nmt.NmtBin.from_nside_linear(self.full_skymap.nside, l_per_bandpower)
        ell = ellbins.get_effective_ells()
        nfields = len(fields)
        nell = len(ell)
        cls = np.empty((nfields, nell))
        for i, field in enumerate(fields):
            cls[i, :] = nmt.compute_full_master(field, field, ellbins)

        return ell, cls

    def auto_cross_cls(self, l_per_bandpower, fields):
        """Generate auto and cross angular power spectra for different fields.

        Parameters
        ----------
        l_per_bandpower : int
            Number of multipoles per bin
        fields : list[nmt.NmtField]
            the list of fields with which to calculate the C_\ell. See `fields` method

        Returns
        -------
        ell: array_like
            Array  of effective multipoles
        cls: array_like
            Array of calculated C_\ell. If ell has shape (N,), this has shape (nfields, nfields, N)
        """
        ellbins = nmt.NmtBin.from_nside_linear(self.full_skymap.nside, l_per_bandpower)
        ell = ellbins.get_effective_ells()
        nfields = len(fields)
        nell = len(ell)
        cls = np.empty((nfields, nfields, nell))
        for i in range(nfields):
            for j in range(i, nfields):
                cls[i, j, :] = nmt.compute_full_master(fields[i], fields[j], ellbins)
                cls[j, i, :] = cls[i, j, :]

        return ell, cls
