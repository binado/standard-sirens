#! /usr/bin/env python3
import argparse
from pathlib import Path
import os
import logging

import numpy as np
from tqdm import tqdm

from ..utils.io import write_to_hdf5
from ..utils.logger import get_logger, DEFAULT_LOGFILE
from ..catalog.parser import GLADECatalogTranslator, GLADECatalogParser


logger = get_logger(__name__, logfile=DEFAULT_LOGFILE)
dirname = os.getcwd()


# Default options
default_chunksize = 200000  # Row chunk size in pandas.read_csv
default_nside = 32  # nside HEALPIX parameter

dtypes = GLADECatalogTranslator.dtypes()
luminosity_bands = GLADECatalogTranslator.luminosity_bands
available_bands = GLADECatalogTranslator.available_bands()

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(prog="parse_catalog", description="Parse GLADE+ catalog")
argument_parser.add_argument("filename", type=Path, help="Path to GLADE+ text file")
argument_parser.add_argument("-o", "--output", default="output.hdf5", help="Output file name")
argument_parser.add_argument("--nside", type=int, default=default_nside, help="nside HEALPIX parameter")
argument_parser.add_argument(
    "-n",
    "--nrows",
    default=None,
    type=int,
    help="Number of catalog rows to read, useful for debugging",
)
argument_parser.add_argument(
    "--catalog-flags", action="store_true", help="Wether to include catalog flags in output file"
)
argument_parser.add_argument(
    "-b",
    "--bands",
    nargs="*",
    help=f"Bands to include in the output file. Available bands: {available_bands}",
)
argument_parser.add_argument(
    "-m", "--mass", action="store_true", help="Wether to include mass and merger rate estimates in output file"
)
argument_parser.add_argument("--nest", action="store_true", help="nest parameter when pixelizing with HEALPIX")
argument_parser.add_argument(
    "-c",
    "--chunksize",
    default=default_chunksize,
    type=int,
    help="Row chunk size in pandas.read_csv",
)
argument_parser.add_argument("--compression", default=None, help="Compression format for output data")
argument_parser.add_argument("-v", "--verbose", action="store_true")


def filter_chunk(df):
    # Remove galaxies with non-positive redshift
    df = df[df["z_cmb"].notnull() & (df["z_cmb"] >= 0)]
    # Remove quasars or clusters
    df = df[df["object_type_flag"] == "G"]
    # Remove galaxies with no redshift or redshift calculated from dl
    df = df[(df["dist_flag"] == 1) | (df["dist_flag"] == 3)]
    # Remove close by galaxies without peculiar velocity corrections
    df = df[~((df["z_flag"] == 0) & (df["z_cmb"] < 0.05))]
    # Remove galaxies with no mass
    # df = df[df["mass"].notnull()]
    return df


def main():
    # Parse command line args
    args = argument_parser.parse_args()
    chunksize = args.chunksize
    verbose = args.verbose
    nside = args.nside
    nrows = args.nrows
    nest = args.nest
    filename = os.path.join(dirname, args.filename)
    output = args.output

    cols = GLADECatalogTranslator.get_columns(args.bands, catalog_flags=args.catalog_flags, mass=args.mass)

    if verbose:
        logging.info("Selected bands for parsing: %s", args.bands)
        logging.info("Selected compression scheme for output data: %s", args.compression)
        logging.info("Starting catalog parsingsirens..")
        logging.info("Reader chunk size: %s", chunksize)
        if nrows is not None:
            logging.info("Parsing the first %s galaxies", nrows)

    catalog = GLADECatalogParser.parse(
        filename, cols, filter_fn=filter_chunk, chunksize=chunksize, nrows=nrows, progress=tqdm
    )
    if verbose:
        logging.info("Catalog parsed successfully with %s objects.", catalog.shape[0])

    # Extract data from catalog
    # (theta, phi) = (ra * 180 / pi + pi/2, dec * 180 / pi)
    catalog["ra"] *= np.pi / 180
    catalog["dec"] *= np.pi / 180

    # Build output file
    logging.info("Writing to output file")
    output_data = {key: data for key, data in catalog.items()}
    write_to_hdf5(output, output_data, dtypes, attrs=dict(nside=nside, nest=nest), compression=args.compression)
    logging.info("Done!")
