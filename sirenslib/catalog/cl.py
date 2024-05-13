import numpy as np
import pymaster as nmt

from .utils import Skymap


class AngularPowerSpectrumEstimator:
    def __init__(self, full_skymap: Skymap, bin_skymaps: list[Skymap], include_full_skymap=False) -> None:
        self.full_skymap = full_skymap
        self.full_map = full_skymap.counts()
        self.bin_skymaps = bin_skymaps
        self.maps = [skymap.counts() for skymap in bin_skymaps]
        if include_full_skymap:
            self.maps.append(full_skymap.counts())

    def mask(self, quantile_cut: float, aposize: float = None, **kwargs):
        if quantile_cut < 0 or quantile_cut > 1:
            raise ValueError

        mask = self.full_map >= np.quantile(self.full_map, quantile_cut)
        if aposize is not None:
            mask = nmt.mask_apodization(mask, aposize=aposize, **kwargs)
        return mask

    def fields(self, mask):
        return [nmt.NmtField(mask, [mapi]) for mapi in self.maps]

    def auto_cls(self, l_per_bandpower, fields: list[nmt.NmtField]):
        ellbins = nmt.NmtBin.from_nside_linear(self.full_skymap.nside, l_per_bandpower)
        ell = ellbins.get_effective_ells()
        nfields = len(fields)
        nell = len(ell)
        cls = np.empty((nfields, nell))
        for i, field in enumerate(fields):
            cls[i, :] = nmt.compute_full_master(field, field, ellbins)

        return ell, cls

    def auto_cross_cls(self, l_per_bandpower, fields: list[nmt.NmtField]):
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
