# Scripts

## `get_catalog`

Bash script to download the GLADE+ catalog.

## `sirenslib-parse_catalog`

Parse the `GLADE+` catalog onto a `hdf5` file - see the [catalog directory](../catalog/README.md) for mor details).

## `sirenslib-dark_siren_inference`

Run MCMC on cosmological and merger rate parameters according to the likelihood model implemented in the [`DrawnGWMergerRatePriorInference` class](../inference/likelihood.py). The chain is stored on an `hdf5` file. Use the [`visualize_mcmc` notebook](../../notebooksvisualize_mcmc.ipynb) to analyse the results.

## `sirenslib-cls`

Compute auto and cross angular power spectra for different redshift bins in a given galaxy catalog, writing the $c_\ell$ to a `hdf5` file. Currently implemented for the `GLADE+` catalog only. .
Uses the `namaster` code for pseudo-$c_\ell$ estimation (see [paper](https://arxiv.org/abs/1809.09603) and [repo](https://github.com/LSSTDESC/NaMaster)).

Here is an example with typical options:

- 5 redshift bins: $0 < z_1 < 0.1 < z_2 < 0.2 < z_3 < 0.3 < z_4 < 0.4 < z_5 < 0.5$
- HEALPIX `nside` = 256
- Apodization scale of 1deg
- Auto mask by removing the 30% least dense pixels (which corresponds roughly to the galactic foreground in `GLADE+`)
- Linear multipole binning parameter `nlb` = 4 (see `namaster` [documentation](https://namaster.readthedocs.io/en/latest/pymaster.html#pymaster.bins.NmtBin.from_nside_linear))
- Save figures to a default directory (`figures/cls/`)
- Verbose output

```bash
sirenslib-cls path_to_parsed_catalog.hdf5 0.0 0.1 0.2 0.3 0.4 --zmax 0.5 --nside 256 -a 1.0 -l 4 -q 0.3 -s -v
```

## `sirenslib-cls_gw`

Compute angular power spectra for the sky distribution of GW sources in the GWTC3 catalog. Uses either `anafast` or `namaster` routines.

Example:

- HEALPIX `nside` = 256
- Maximum multipole `lmax` = 100
- `anafast` routine for estimating the power spectra

```bash
sirenslib-cls_gw data/gwtc3-localizations/ -s --nside 256 --lmax 100 -m anafast
```
