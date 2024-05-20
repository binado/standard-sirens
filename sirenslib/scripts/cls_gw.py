import argparse
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from ..gw.skymap import get_skymaps, GWSkymap, get_combined_skymap

parser = argparse.ArgumentParser(description="Estimate the angular power spectrum of GW skymaps.")

parser.add_argument("dir", type=str, help="Path to localization directory")
parser.add_argument("--nside", type=int, default=256, help="nside HEALPIX parameter")
parser.add_argument("--figure-dir", default="figures/cls_gw")
parser.add_argument("-s", "--save-figure", action="store_true", help="Save output to figure")
parser.add_argument("--figure-format", type=str, default="png", help="Figure format")
parser.add_argument("--lmin", type=int, default=2, help="Minimum l for plotting")
parser.add_argument("--lmax", type=int, default=None, help="Maximum l for plotting")
parser.add_argument("-v", "--verbose", action="store_true")


def main():
    args = parser.parse_args()

    base_dir = Path(args.dir)
    csv_file = list(base_dir.glob("*.csv"))[0]

    if args.save_figure:
        out_dir = Path(args.figure_dir)
        out_dir.mkdir(exist_ok=True)

    gw_skymaps = []
    for probmap in get_skymaps(base_dir, csv_file):
        gw_skymaps.append(GWSkymap(probmap))

    combined_skymap = get_combined_skymap(gw_skymaps, args.nside)
    cls = combined_skymap.power_spectrum(lmax=args.lmax)
    ells = np.arange(1, cls.size + 1)

    if args.save_figure:
        # Plotting
        hp.mollview(combined_skymap.probmap, norm="log", coord=["E", "G"], title="GW Skymap", cmap="viridis")
        figpath = out_dir / f"skymap_gw_nside={args.nside}.{args.figure_format}"
        plt.savefig(figpath, dpi=400)

        fig, ax = plt.subplots()
        ax.semilogy(ells[2:], cls[2:])
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell / C_0$")
        ax.grid()
        fig.tight_layout()
        figpath = out_dir / f"cls_gw_nside={args.nside}.{args.figure_format}"
        fig.savefig(figpath, dpi=400)
