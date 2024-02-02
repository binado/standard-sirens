import numpy as np
import pandas as pd
from tqdm import tqdm


class GLADECatalogTranslator:
    catalog_flags = {
        "GLADE_no": str,
        "PGC_no": str,
        "GWGC_name": str,
        "Hyperleda_name": str,
        "2MASS_name": str,
        "WISExSCOS_name": str,
        "SDSS-DR16Q_name": str,
    }
    quasar_flag = {
        "object_type_flag": str,
    }
    loc = {
        "ra": np.float64,
        "dec": np.float64,
    }
    luminosity_bands = {
        "B": {
            "m_B": np.float64,
            "m_B_err": np.float64,
            "m_B_flag": np.float64,
            "m_B_abs": np.float64,
        },
        "J": {
            "m_J": np.float64,
            "m_J_err": np.float64,
        },
        "H": {
            "m_H": np.float64,
            "m_H_err": np.float64,
        },
        "K": {
            "m_K": np.float64,
            "m_K_err": np.float64,
        },
        "W1": {
            "m_W1": np.float64,
            "m_W1_err": np.float64,
        },
        "W2": {"m_W2": np.float64, "m_W2_err": np.float64, "m_W1_flag": np.float64},
        "BJ": {
            "m_BJ": np.float64,
            "m_BJ_err": np.float64,
        },
    }
    redshift = {
        "z_helio": np.float64,
        "z_cmb": np.float64,
        "z_flag": np.float64,
        "v_err": np.float64,
        "z_err": np.float64,
        "d_L": np.float64,
        "d_L_err": np.float64,
        "dist_flag": np.float64,
    }
    mass_and_merger_rate = {
        "m_*": np.float64,
        "m_*_err": np.float64,
        "m_*_flag": np.float64,
        "merger_rate": np.float64,
        "merger_rate_err": np.float64,
    }

    @classmethod
    def available_bands(cls):
        return list(cls.luminosity_bands.keys())

    @classmethod
    def luminosity_dtypes(cls):
        luminosity_dtypes = dict()
        for flags in cls.luminosity_bands.values():
            luminosity_dtypes.update(**flags)
        return luminosity_dtypes

    @classmethod
    def dtypes(cls):
        luminosity_dtypes = cls.luminosity_dtypes()
        return {
            **cls.catalog_flags,
            **cls.quasar_flag,
            **cls.loc,
            **luminosity_dtypes,
            **cls.redshift,
            **cls.mass_and_merger_rate,
        }

    @classmethod
    def available_columns(cls):
        return list(cls.dtypes().keys())

    @classmethod
    def get_columns(cls, bands, catalog_flags=True, quasar_flag=True, loc=True, redshift=True, mass=True):
        available_bands = cls.available_bands()
        columns = []
        if catalog_flags:
            columns += list(cls.catalog_flags.keys())
        if quasar_flag:
            columns += list(cls.quasar_flag.keys())
        if loc:
            columns += list(cls.loc.keys())
        if redshift:
            columns += list(cls.redshift.keys())
        if mass:
            columns += list(cls.mass_and_merger_rate.keys())
        for band in bands:
            if band in available_bands:
                columns += list(cls.luminosity_bands[band].keys())

        return columns


class GLADECatalogParser:
    @staticmethod
    def parse(file, cols, filter_fn, chunksize=200000, progress=tqdm, **kwargs):
        """Parse the GlADE+ text file into a Pandas DataFrame.

        Uses Pandas.read_csv method.

        Parameters
        ----------
        filename : str
            The path to the GLADE+ text file
        cols : list
            The list of columns to extract from the file. See `GlADECatalogTranslator.get_columns`
        filter_fn : function
            A filter function to be executed on each DataFrame chunk
        chunksize : int, optional
            The chunksize argument of read_csv. Defaults to 200000, which corresponds to roughly 100 iterations
        progress :  optional
            A progress bar decorator, by default tqdm

        Returns
        -------
        Pandas.DataFrame
            The DataFrame containing the extracted columns after filtering out the data
        """
        dtypes = GLADECatalogTranslator.dtypes()
        reader_args = dict(
            sep=" ",
            names=dtypes.keys(),
            usecols=cols,
            dtype=dtypes,
            header=None,
            false_values=["null"],
            chunksize=chunksize,
            **kwargs
        )
        chunks = []
        with pd.read_csv(file, **reader_args) as reader:
            iterator = progress(reader) if progress else reader
            for chunk in iterator:
                chunks.append(filter_fn(chunk))

            catalog = pd.concat(chunks, ignore_index=True)

        return catalog
