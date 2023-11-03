import pandas as pd
import numpy as np
import healpy as hp
import h5py
import argparse
from pathlib import Path
import os


dirname = os.getcwd()


# Default options
default_chunksize = 100000  # Row chunk size in pandas.read_csv
chunk_step_progress_message = 10
default_nside = 1024  # nside HEALPIX parameter

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(
    prog="parse_catalog", description="Parse GLADE+ catalog"
)
argument_parser.add_argument("filename", type=Path)
argument_parser.add_argument("-o", "--output", default="output.hdf5")
argument_parser.add_argument(
    "--nside", type=int, default=default_nside, help="nside HEALPIX parameter"
)
argument_parser.add_argument(
    "-n",
    "--nrows",
    default=None,
    type=int,
    help="Number of catalog rows to read, useful for debugging",
)
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
    return df


def create_or_overwrite_dataset(file, dataset, data):
    try:
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

        for index, chunk in enumerate(reader):
            # print(f"chunk has {chunk.shape[0]} rows")
            catalog = pd.concat([catalog, filter_chunk(chunk)], ignore_index=True)
            if verbose and (index + 1) % chunk_step_progress_message == 0:
                print(f"Parsed chunk number {index + 1}")
                print(f"Parsed catalog has {catalog.shape[0]} rows")
                print(f"{chunksize * (index + 1) - catalog.shape[0]} rows filtered out")
                print(
                    "{:.1f}% rows included".format(
                        100 * catalog.shape[0] / chunksize / (index + 1)
                    )
                )
                print("-------------------------------------------")

    # Extract data from catalog
    # (theta, phi) = (ra * 180 / pi + pi/2, dec * 180 / pi)
    ra = catalog["ra"] * np.pi / 180
    dec = catalog["dec"] * np.pi / 180
    skymap = hp.ang2pix(nside, dec + np.pi / 2, ra)
    z = catalog["z_cmb"]

    # Remove output file if it already exists
    # Hack to prevent OSError
    # if os.path.exists(output):
    #     os.remove(output)

    # Build output file
    with h5py.File(output, "a") as f:
        if verbose:
            print("Creating output datasets...")

        create_or_overwrite_dataset(f, "ra", data=ra)
        create_or_overwrite_dataset(f, "dec", data=dec)
        create_or_overwrite_dataset(f, "skymap", data=skymap)
        create_or_overwrite_dataset(f, "z", data=z)
        f.attrs["nside"] = nside

    print("Done!")
