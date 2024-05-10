#! /usr/bin/env python3
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
argument_parser.add_argument("-b", "--bins", type=float, nargs="*", default=None, help="Custom redshift bins")
argument_parser.add_argument("-n", "--nbins", type=int, default=5, help="Number of bins")
argument_parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift")
argument_parser.add_argument("--zmax", type=float, default=0.5, help="Maximum redshift")
argument_parser.add_argument("-o", "--output-dir", default="data/cls", help="Output file directory")
argument_parser.add_argument("-c", "--cross", action="store_true", help="Compute cross spectra as well")
argument_parser.add_argument("--nside", type=int, default=64, help="nside HEALPIX parameter")
argument_parser.add_argument("-a", "--aposize", type=float, default=None, help="Mask apodization scale")
argument_parser.add_argument("-l", "--l-per-bandpower", type=int, default=4, help="l per bandpower setting in namaster")
argument_parser.add_argument("-q", "--quantile", type=float, default=0.25, help="Discard q-lowest density pixels")
argument_parser.add_argument("--figure-dir", default="figures/cls")
argument_parser.add_argument("-s", "--save-figure", action="store_true", help="Save output to figure (only auto cls)")
argument_parser.add_argument("--figure-format", type=str, default="png", help="Figure format")
argument_parser.add_argument("--lmin", type=int, default=2, help="Minimum l for plotting")
argument_parser.add_argument("--lmax", type=int, default=-1, help="Maximum l for plotting")
argument_parser.add_argument("-v", "--verbose", action="store_true")


def main():
    args = argument_parser.parse_args()
    v = args.verbose
    if v:
        logging.info("Starting catalog reading")

    nbins = len(args.bins) if args.bins is not None else args.nbins

    filename = f"GLADE+_nbins={nbins}_nside={args.nside}_aposize={args.aposize}_cross={args.cross}"
    output_filename = f"{args.output_dir}/{filename}.hdf5"

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
    if args.bins is not None:
        zbins = np.array(args.bins)
    else:
        zbins = np.linspace(args.zmin, args.zmax, args.nbins + 1)
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

    # Plotting
    if args.save_figure:
        import os
        import matplotlib.pyplot as plt

        figure_dir = f"{args.figure_dir}/{filename}"
        if not os.path.isdir(figure_dir):
            os.mkdir(figure_dir)

        titles = [r"${:.2f} < z < {:.2f}$".format(zbins[i], zbins[i + 1]) for i in range(nbins - 1)]
        titles.append(r"$z > {:.2f}$".format(zbins[nbins - 1]))
        lmin, lmax = args.lmin, args.lmax

        # Vertical plots
        fig, axs = plt.subplots(nbins, 1, figsize=(10, 20))
        for i, ax in enumerate(axs):
            y = cls[i, i, lmin:lmax] if args.cross else cls[i, lmin:lmax]
            ax.plot(ell[lmin:lmax], y)
            ax.set_title(titles[i])
            ax.set_ylabel(r"$c_\ell$")
            ax.set_xlabel(r"$\ell$")
            ax.set_yscale("log")
            ax.grid()

        fig.tight_layout()
        fig.savefig(f"{figure_dir}/vertical.{args.figure_format}", dpi=400)

        # Single plot
        fig, ax = plt.subplots()
        for i in range(nbins):
            y = cls[i, i, lmin:lmax] if args.cross else cls[i, lmin:lmax]
            ax.plot(ell[lmin:lmax], y, label=titles[i])
        ax.set_ylabel(r"$c_\ell$")
        ax.set_xlabel(r"$\ell$")
        ax.set_yscale("log")
        ax.legend()
        ax.grid()

        fig.tight_layout()
        fig.savefig(f"{figure_dir}/single.{args.figure_format}", dpi=400)

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
