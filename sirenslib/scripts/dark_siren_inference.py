#! /usr/bin/env python3
import os
import logging
import argparse
from pprint import pformat
from uuid import uuid4

import numpy as np

from ..catalog.utils import GalaxyCatalog
from ..gw.event import GWEventGenerator
from ..inference.likelihood import DrawnGWPopulationInference
from ..inference.population import MadauDickinsonRedshiftPrior
from ..inference.prior import Parameters, UniformPrior
from ..inference.sampling import MCMCStrategy, NestedStrategy, SamplingRun, SAMPLING_STRATEGIES
from ..utils.logger import logging_config
from ..utils.functions import list_to_str
from ..utils.math import sample_from_func
from ..utils.cosmology import luminosity_distance, flat_cosmology

dirname = os.getcwd()

# Default arguments
# theta = {alpha, beta, c}
labels = ["H0", "alpha", "beta", "c"]
plot_labels = [r"$H_0$", r"$\alpha$", r"$\beta$", r"$c$"]
fiducial_H0 = 70
md_theta = (2.7, 2.9, 1.9)
truths = [fiducial_H0, *md_theta]
prior_min, prior_max = np.array([20.0, 0.0, 0.0, 0.0]), np.array([140.0, 10.0, 10.0, 4.0])

# Catalog parameters
n_dir = 5
n_min = 100
alpha = np.radians(2)
catalog_filename = "./catalog/output.hdf5"
catalog = GalaxyCatalog(catalog_filename)

# Sampling parameters
nevents = 200
sigma_dl = 0.1
z_th = 10
n_walkers = 32
nsteps = 10000
full_z = np.linspace(1e-4, 10.0, 2000)

logger_output_file = "logs/scripts.log"
output_dir = "data/runs"
samples_dir = "data/samples"
output_group = "mcmc"

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(
    prog="inference_merger_rate_prior", description="Run bayesian inference on cosmological and merger rate parameters."
)
argument_parser.add_argument(
    "-o", "--output", type=str, default=output_dir, help="Path to output directory of the sampling data"
)
argument_parser.add_argument("-g", "--group", type=str, default=output_group, help="hdf5 file group to store MCMC data")
argument_parser.add_argument(
    "-p", "--params", type=str, nargs="*", default=labels, choices=labels, help="Parameters to infer"
)
argument_parser.add_argument(
    "-t",
    "--truths",
    type=np.float64,
    nargs="*",
    default=truths,
    help="Fiducial values of the parameters for simulations",
)
argument_parser.add_argument(
    "--sigmadl", type=np.float64, default=sigma_dl, help="Luminosity distance uncertainty multiplier"
)
argument_parser.add_argument("--low-redshift", action="store_true", default=False)
argument_parser.add_argument("--nwalkers", type=int, default=n_walkers, help="Number of walkers in ensemble sampler")
argument_parser.add_argument("-n", "--nsteps", type=int, default=nsteps, help="Number of MCMC steps")
argument_parser.add_argument("-e", "--events", type=int, default=nevents, help="Number of injections to simulate")
argument_parser.add_argument(
    "--strategy",
    default="mcmc",
    choices=SAMPLING_STRATEGIES.keys(),
    help="Use 'mcmc' or 'nested' for different sampling methods",
)
argument_parser.add_argument("-s", "--silent", action="store_true", default=False)


def main():
    args = argument_parser.parse_args()
    output_group = args.group
    verbose = not args.silent
    sigma_dl = args.sigmadl
    low_redshift = args.low_redshift
    nevents = args.events
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    truths = args.truths

    dataformat = SAMPLING_STRATEGIES[args.strategy].get("format")
    output_filename = (
        f"{args.output}/{list_to_str(args.params)}_strategy={args.strategy}_nevents={nevents}_nsteps={nsteps}.json"
    )
    samples_filename = str(uuid4())
    samples_filepath = f"{samples_dir}/{samples_filename}.{dataformat}"

    n_gw_per_dir = np.ceil(nevents / n_dir)
    fiducial_H0, *theta = truths
    fiducial_cosmology = flat_cosmology(fiducial_H0)
    dl_th = luminosity_distance(fiducial_cosmology, z_th)

    # Metadata to be recorded at the end of the run
    attrs = {
        "strategy": SAMPLING_STRATEGIES[args.strategy].get("name"),
        "samples_filepath": samples_filepath,
        "sigma_dl": sigma_dl,
        "z_th": z_th,
        "dl_th": dl_th,
        "nevents": nevents,
        "nwalkers": nwalkers,
        "nsteps": nsteps,
    }

    # Get params which remain fixed to fiducial values
    fixed_params = {label: truth for label, truth in zip(labels, truths) if label not in args.params}

    # Construct uniform prior for free params
    mask = [label not in args.params for label in labels]
    pmin = np.ma.array(prior_min, mask=mask).compressed()
    pmax = np.ma.array(prior_max, mask=mask).compressed()
    prior = UniformPrior(pmin, pmax)

    # Define params to be used
    params = Parameters(labels, plot_labels, truths, **fixed_params)

    logging_config(logger_output_file)
    if verbose:
        logging.info("Starting inference run for parameters %s with:", args.params)
        logging.info(pformat(attrs))

    # Generating events
    z_prior = MadauDickinsonRedshiftPrior(fiducial_cosmology, full_z)
    z_sample = sample_from_func(100000 * nevents, z_prior.eval, full_z, *theta, normalize=True)
    events = GWEventGenerator(dl_th).from_redshifts(fiducial_cosmology, z_sample, sigma_dl)[:nevents]

    if verbose:
        logging.info("Mean generated redshift: %s", np.average(z_sample))
        logging.info("Median generated redshift: %s", np.median(z_sample))
    if verbose:
        logging.info("%s events were generated", len(events))

    inference = DrawnGWPopulationInference(
        z_prior, events, params, prior, fiducial_H0=fiducial_H0, sigma_dl=sigma_dl, dl_th=dl_th
    )
    # cosmology = inference.fiducial_cosmology
    # Draw galaxy redshifts and masses in n_dir directions, with at least n_min per direction
    # drawn_galaxies = catalog.draw_galaxies(n_dir, alpha, n_min)
    # z_gal = [catalog.z[galaxies_at_direction] for galaxies_at_direction in drawn_galaxies]
    # mass_gal = [catalog.mass[galaxies_at_direction] for galaxies_at_direction in drawn_galaxies]
    # generator = EventGenerator(fiducial_H0, dl_th=dl_th)
    # events = generator.from_catalog(cosmology, z_gal, sigma_dl, n_gw_per_dir)

    ndim = prior.ndim
    sampling_args = []
    sampling_kwargs = {}
    # Initialize sampling strategy
    if args.strategy == "mcmc":
        strategy = MCMCStrategy(
            nwalkers, ndim, inference.log_posterior, args=[events, full_z], filename=samples_filepath, reset=True
        )
        # Initialize walkers around fiducial values + normal fluctuations
        var = 1e-1 * prior.interval
        initial = strategy.initial_around(np.ma.array(truths, mask=mask).compressed(), var, prior)
        sampling_args += [initial, nsteps]
        sampling_kwargs.update(progress=verbose, store=True)
    else:
        strategy = NestedStrategy(inference.log_likelihood, prior, filename=samples_filepath)
        sampling_kwargs.update(maxiter=nsteps)

    if verbose:
        logging.info("Starting %s", SAMPLING_STRATEGIES[args.strategy])
    strategy.run(*sampling_args, **sampling_kwargs)
    logging.info("Saving run summary to %s", output_filename)
    sampling_run = SamplingRun(params, prior, strategy)
    sampling_run.save_to_json(output_filename, **attrs)
