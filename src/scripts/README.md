# Scripts

## `get_catalog`

Bash script to download the GLADE+ catalog.

## `parse_catalog`

Parse the `GLADE+` catalog onto a `hdf5` file - see the [catalog directory](../catalog/README.md) for mor details).

## `inference_merger_rate_prior`

Run MCMC on cosmological and merger rate parameters according to the likelihood model implemented in the [`DrawnGWMergerRatePriorInference` class](../inference/likelihood.py). The chain is stored on an `hdf5` file. Use the [`visualize_mcmc` notebook](../visualize_mcmc.ipynb) to analyse the results.

## `cls`

Compute auto and cross angular power spectra for different redshift bins in a given galaxy catalog. Currently implemented for the `GLADE+` catalog only. Outputs $c_\ell$ to a `hdf5` file.
