import argparse
from pathlib import Path
import os
import logging

import healpy as hp
import numpy as np

from ..catalog.utils import GalaxyCatalog, Skymap
from ..utils.io import write_to_hdf5
from ..utils.logger import logging_config


# Prevent bug with healpy.anafast
# Read more here: https://github.com/dmlc/xgboost/issues/1715
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logging_config("logs/scripts.log")

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(
    prog="catalog_angular_power_spectrum", description="Compute catalog angular power spectrum"
)
argument_parser.add_argument("filename", type=Path, help="Path to GLADE+ text file")
argument_parser.add_argument("bins", type=float, nargs="*", help="Redshift bins")
argument_parser.add_argument("-o", "--output", default="output.hdf5", help="Output file name")
argument_parser.add_argument("--nside", type=int, default=64, help="nside HEALPIX parameter")
argument_parser.add_argument("--compression", default=None, help="Compression format for output data")
argument_parser.add_argument("-v", "--verbose", action="store_true")

if __name__ == "__main__":
    args = argument_parser.parse_args()
    v = args.verbose
    if v:
        logging.info("Starting catalog reading")

    catalog = GalaxyCatalog(args.filename)
    ra, dec = catalog.get("ra"), catalog.get("dec")
    z = catalog.get("z_cmb")
    if v:
        logging.info("Done!")
        logging.info("Splitting data into different redshift bins")

    # Create masks for each redshift bin
    z_bins = np.array(args.bins)
    masks = Skymap.bin_array(z, z_bins)
    # Create separate (ra, dec) arrays for each redshift bin
    ra_per_bin = [Skymap.remove_mask(ra, ~mask) for mask in masks]
    dec_per_bin = [Skymap.remove_mask(dec, ~mask) for mask in masks]
    if v:
        logging.info("Done!")
        logging.info("Computing the angular power spectrum for each bin")

    lmax = 3 * args.nside - 1
    cls_per_bin = {}
    dtypes = {}
    for i, (rabin, decbin) in enumerate(zip(ra_per_bin, dec_per_bin)):
        bin_skymap = Skymap(args.nside, rabin, decbin, nest=False)
        counts_in_shell = bin_skymap.counts()
        cls = hp.anafast(counts_in_shell, lmax=lmax)
        cls_per_bin[f"bin_{i}"] = cls
        dtypes[f"bin_{i}"] = np.float64
    if v:
        logging.info("Done!")
        logging.info("Writing to output file")

    write_to_hdf5(args.output, cls_per_bin, dtypes, prefix="bins/")
    write_to_hdf5(args.output, dict(bins=z_bins), dict(bins=np.float64))
    if v:
        logging.info("Done!")
