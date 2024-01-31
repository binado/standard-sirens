import os
import logging
import argparse
from pathlib import Path

import numpy as np
from emcee import EnsembleSampler
from emcee.backends import HDFBackend

from ..inference.likelihood import DrawnGWMergerRatePriorInference
from ..inference.utils import merger_rate, low_redshift_merger_rate, sample_from_func, EventGenerator
from ..inference.prior import UniformPrior
from ..utils.logger import logging_config

dirname = os.getcwd()

# Default arguments
# theta = {alpha, beta, c}
prior_min, prior_max = np.array([20.0, -10.0, 0.0, 1.0]), np.array([140.0, 10.0, 10.0, 10.0])
madau_like_prior = UniformPrior(prior_min, prior_max)
low_redshift_prior = UniformPrior(prior_min[:2], prior_max[:2])

fiducial_H0 = 70
md_theta = [2.7, 5.6, 2.9]
H0_min, H0_max = 20, 140
n_events = 200
sigma_dl = 0.1
n_walkers = 32
n_steps = 1000

logger_output_file = "logs/scripts.log"
output_filename = "output.hdf5"
output_group = "mcmc"

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(
    prog="inference_merger_rate_prior", description="Run bayesian inference on cosmological and merger rate parameters."
)
argument_parser.add_argument("-o", "--output", type=Path, default=output_filename, help="Path to output the MCMC data")
argument_parser.add_argument("-g", "--group", type=str, default=output_group, help="hdf5 file group to store MCMC data")
argument_parser.add_argument(
    "--H0", type=np.float64, default=fiducial_H0, help="Fiducial value of H0 to generate injections"
)
argument_parser.add_argument(
    "--sigmadl", type=np.float64, default=sigma_dl, help="Luminosity distance uncertainty multiplier"
)
argument_parser.add_argument("--low-redshift", action="store_true", default=False)
argument_parser.add_argument("--nwalkers", type=int, default=n_walkers, help="Number of walkers in ensemble sampler")
argument_parser.add_argument("-n", "--nsteps", type=int, default=n_steps, help="Number of MCMC steps")
argument_parser.add_argument("-e", "--events", type=int, default=n_events, help="Number of injections to simulate")
argument_parser.add_argument("-s", "--silent", action="store_true", default=False)


if __name__ == "__main__":
    args = argument_parser.parse_args()
    output_filename = args.output
    output_group = args.group
    verbose = not args.silent
    fiducial_H0 = args.H0
    sigma_dl = args.sigmadl
    low_redshift = args.low_redshift
    n_events = args.events
    n_walkers = args.nwalkers
    n_steps = args.nsteps
    fiducial = np.array([fiducial_H0, *md_theta])

    n_theta = 1 if low_redshift else len(md_theta)
    n_params = n_theta + 1  # Theta + H0

    logging_config(logger_output_file)
    if verbose:
        logging.info(
            "Started inference run with H0=%s, sigma_dl=%s with %s injections", fiducial_H0, sigma_dl, n_events
        )
        logging.info("Low redshift configuration: %s", low_redshift)

    full_z = np.linspace(1e-4, 20, 1000)
    mr = low_redshift_merger_rate if low_redshift else merger_rate
    event_redshifts = sample_from_func(100 * n_events, merger_rate, full_z, *md_theta)
    if verbose:
        logging.info("Mean generated redshift: %s", np.average(event_redshifts))
        logging.info("Median generated redshift: %s", np.median(event_redshifts))
    prior = low_redshift_prior if low_redshift else madau_like_prior
    inference = DrawnGWMergerRatePriorInference(
        full_z,
        prior,
        fiducial_H0=fiducial_H0,
        sigma_dl=sigma_dl,
        low_redshift=low_redshift,
    )
    cosmology = inference.fiducial_cosmology
    events = EventGenerator(fiducial_H0).from_redshifts(cosmology, event_redshifts, sigma_dl)[:n_events]
    if verbose:
        logging.info("%s events were generated", len(events))

    # Initialize walkers around fiducial values + normal fluctuations
    initial = fiducial[:n_params] + 1e-1 * prior.interval * np.random.randn(32, n_params)
    assert np.all(
        np.isfinite([prior.log_prior(walker) for walker in initial])
    ), "Initial state not within allowed prior range"
    nwalkers, ndim = initial.shape

    backend = HDFBackend(output_filename, name=output_group)
    backend.reset(nwalkers, ndim)
    sampler = EnsembleSampler(nwalkers, ndim, inference.log_posterior, args=[events, full_z], backend=backend)
    if verbose:
        logging.info("Starting MCMC")
    sampler.run_mcmc(initial, n_steps, progress=verbose, store=True)
