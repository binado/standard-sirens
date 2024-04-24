import argparse
from pathlib import Path
import logging

import h5py
import numpy as np
import pymaster as nmt

from ..catalog.utils import GalaxyCatalog, Skymap, MaskedMap
from ..catalog.cl import AngularPowerSpectrumEstimator
from ..utils.io import create_or_overwrite_dataset
from ..utils.logger import logging_config


logging_config("logs/scripts.log")

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(
    prog="catalog_angular_power_spectrum", description="Compute catalog angular power spectrum"
)
argument_parser.add_argument("filename", type=Path, help="Path to GLADE+ text file")
argument_parser.add_argument("bins", type=float, nargs="*", help="Redshift bins")
argument_parser.add_argument("--zmax", type=float, default=0.5, help="Maximum redshift")
argument_parser.add_argument("-o", "--output-dir", default="data/cls", help="Output file directory")
argument_parser.add_argument("-c", "--cross", action="store_true", help="Compute cross spectra as well")
argument_parser.add_argument("--nside", type=int, default=64, help="nside HEALPIX parameter")
argument_parser.add_argument("-a", "--aposize", type=float, default=None, help="Mask apodization scale")
argument_parser.add_argument("-l", "--l-per-bandpower", type=int, default=4, help="l per bandpower setting in namaster")
argument_parser.add_argument("-q", "--quantile", type=float, default=0.25, help="Discard q-lowest density pixels")
argument_parser.add_argument("-v", "--verbose", action="store_true")

if __name__ == "__main__":
    args = argument_parser.parse_args()
    v = args.verbose
    if v:
        logging.info("Starting catalog reading")

    output_filename = (
        f"{args.output_dir}/GLADE+_bins={args.bins}_nside={args.nside}_aposize={args.aposize}_cross={args.cross}.hdf5"
    )

    catalog = GalaxyCatalog(args.filename)
    ra, dec = catalog.get("ra"), catalog.get("dec")
    z = catalog.get("z_cmb")
    if v:
        logging.info("Done!")
        logging.info("Splitting data into different redshift bins")

    # Create mask removing the 100q% lowest density pixels
    zmax_mask = z <= args.zmax
    ra_lowz = MaskedMap(ra, masks=[zmax_mask]).compressed
    dec_lowz = MaskedMap(dec, masks=[zmax_mask]).compressed
    full_skymap = Skymap(args.nside, ra_lowz, dec_lowz, nest=False)
    masked_full_map = MaskedMap(full_skymap.counts(), q=args.quantile)
    apodized_mask = nmt.mask_apodization(masked_full_map.mask, args.aposize)

    # Create masks for each redshift bin
    zbins = np.array(args.bins)
    binmasks = Skymap.bin_array(z, zbins)
    # Create separate (ra, dec) arrays for each redshift bin
    bin_skymaps = []
    for binmask in binmasks:
        ra_map = MaskedMap(ra, masks=[binmask, zmax_mask])
        dec_map = MaskedMap(dec, masks=[binmask, zmax_mask])
        bin_skymaps.append(Skymap(args.nside, ra_map.compressed, dec_map.compressed, nest=False))
    if v:
        logging.info("Done!")
        logging.info("Computing the angular power spectrum for each bin")

    clest = AngularPowerSpectrumEstimator(full_skymap, bin_skymaps)
    fields = clest.fields(apodized_mask)
    cls_func = clest.auto_cross_cls if args.cross else clest.auto_cls
    ell, cls = cls_func(args.l_per_bandpower, fields)
    if v:
        logging.info("Done!")
        logging.info("Writing to output file")

    with h5py.File(output_filename, "a") as f:
        create_or_overwrite_dataset(f, "ell", ell, dtype=np.float64)
        create_or_overwrite_dataset(f, "cls", cls, dtype=np.float64)
        f.attrs.create("bins", zbins)
        f.attrs.create("nside", args.nside, dtype=int)
        f.attrs.create("zmax", args.zmax)
        f.attrs.create("aposize", args.aposize)
        f.attrs.create("l_per_bandpower", args.l_per_bandpower, dtype=int)
        f.attrs.create("quantiles", args.quantile)
        f.attrs.create("autocross", args.cross, dtype=bool)

    if v:
        logging.info("Done!")
