import pandas as pd
import numpy as np
import healpy as hp
import h5py
import argparse
from pathlib import Path
import os
from tqdm import tqdm


dirname = os.getcwd()


# Default options
default_chunksize = 100000  # Row chunk size in pandas.read_csv
default_nside = 32  # nside HEALPIX parameter

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(prog="parse_catalog", description="Parse GLADE+ catalog")
argument_parser.add_argument("filename", type=Path, help="Catalog file name")
argument_parser.add_argument("-o", "--output", default="output.hdf5", help="Output file name")
argument_parser.add_argument("--nside", type=int, default=default_nside, help="nside HEALPIX parameter")
argument_parser.add_argument(
    "-n",
    "--nrows",
    default=None,
    type=int,
    help="Number of catalog rows to read, useful for debugging",
)
argument_parser.add_argument("--nest", action="store_true", help="nest parameter when pixelizing with HEALPIX")
argument_parser.add_argument(
    "-c",
    "--chunksize",
    default=default_chunksize,
    type=int,
    help="Row chunk size in pandas.read_csv",
)
argument_parser.add_argument("-v", "--verbose", action="store_true")

# Types for each data column
dtypes = {
    "GWGC_flag": str,
    "Hyperleda_flag": str,
    "2MASS_flag": str,
    "WISE_flag": str,
    "SDSS_flag": str,
    "Quasar_flag": str,
    "ra": np.float64,
    "dec": np.float64,
    "m_K": np.float64,
    "m_K_err": np.float64,
    "z_helio": np.float64,
    "z_cmb": np.float64,
    "peculiar_velocity_correction_flag": "Int64",
    "peculiar_velocity_err": np.float64,
    "z_helio_err": np.float64,
    "redshift_dl_flag": "Int64",
    "mass": np.float64,
    "mass_err": np.float64,
}


def filter_chunk(df):
    # Remove galaxies with non-positive redshift
    df = df[df["z_cmb"].notnull() & (df["z_cmb"] >= 0)]
    # Remove quasars or clusters
    df = df[df["Quasar_flag"] == "G"]
    # Remove galaxies with no redshift or redshift calculated from dl
    df = df[(df["redshift_dl_flag"] == 1) | (df["redshift_dl_flag"] == 3)]
    # Remove close by galaxies without peculiar velocity corrections
    df = df[~((df["peculiar_velocity_correction_flag"] == 0) & (df["z_cmb"] < 0.05))]
    # Remove galaxies with no mass
    df = df[df["mass"].notnull()]
    return df


def create_or_overwrite_dataset(file: h5py.File, dataset, data):
    """
    Overwrite existing dataset with new data for a hdf5 file.
    If such dataset does not exist, create one and populate it
    """
    try:
        # See https://stackoverflow.com/a/22925117
        old_data = file[dataset]
        old_data[...] = data
    except KeyError:
        file.create_dataset(dataset, data=data)


if __name__ == "__main__":
    # Parse command line args
    args = argument_parser.parse_args()
    chunksize = args.chunksize
    verbose = args.verbose
    nside = args.nside
    nrows = args.nrows
    nest = args.nest
    filename = os.path.join(dirname, args.filename)
    output = args.output

    reader_args = dict(
        sep=" ",
        names=dtypes.keys(),
        dtype=dtypes,
        header=None,
        false_values=["null"],
        chunksize=chunksize,
        nrows=nrows,
    )

    catalog = pd.DataFrame()
    with pd.read_csv(filename, **reader_args) as reader:
        if verbose:
            print("Starting catalog parsing...")
            print(f"Reader chunk size: {chunksize}")
            if nrows is not None:
                print(f"Parsing the first {nrows} galaxies")

        for index, chunk in tqdm(enumerate(reader)):
            # print(f"chunk has {chunk.shape[0]} rows")
            catalog = pd.concat([catalog, filter_chunk(chunk)], ignore_index=True)
    if verbose:
        print(f"Catalog parsed successfully with {catalog.shape[0]} objects.")

    # Extract data from catalog
    # (theta, phi) = (ra * 180 / pi + pi/2, dec * 180 / pi)
    ra = catalog["ra"] * np.pi / 180
    dec = catalog["dec"] * np.pi / 180
    skymap = hp.ang2pix(nside, dec + np.pi / 2, ra, nest=nest)
    z = catalog["z_cmb"]
    mass = catalog["mass"]

    # Build output file
    with h5py.File(output, "a") as f:
        if verbose:
            print("Creating output datasets...")

        create_or_overwrite_dataset(f, "ra", data=ra)
        create_or_overwrite_dataset(f, "dec", data=dec)
        create_or_overwrite_dataset(f, "skymap", data=skymap)
        create_or_overwrite_dataset(f, "z", data=z)
        create_or_overwrite_dataset(f, "mass", data=mass)
        f.attrs["nside"] = nside
        f.attrs["nest"] = nest

    print("Done!")
