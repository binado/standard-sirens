import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from emcee import EnsembleSampler
from corner import corner

from ..inference.likelihood import DrawnGWMergerRatePriorInference
from ..inference.utils import merger_rate, sample_from_func, EventGenerator

dirname = os.getcwd()

# Default arguments
theta_min, theta_max = [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]
fiducial_H0 = 70
md_theta = [2.7, 5.6, 2.9]
H0_min, H0_max = 20, 140

n_events = 200

# Adding CLI arguments
argument_parser = argparse.ArgumentParser(
    prog="inference_merger_rate_prior", description="Run bayesian inference on cosmological and merger rate parameters."
)
# argument_parser.add_argument("filename", type=Path, help="Path to parsed data in .hdf5 format")
argument_parser.add_argument("-o", "--output", type=Path, default="output.hdf5", help="Path to output data")
argument_parser.add_argument(
    "--H0", type=np.float64, default=fiducial_H0, help="Fiducial value of H0 to generate injections"
)
argument_parser.add_argument("-n", "--nevents", type=int, default=n_events, help="Number of injections to simulate")
argument_parser.add_argument("-v", "--verbose", action="store_true")

if __name__ == "__main__":
    args = argument_parser.parse_args()
    verbose = args.verbose
    fiducial_H0 = args.H0
    n_events = args.nevents
    fiducial = [fiducial_H0, *md_theta]

    full_z = np.linspace(0, 20, 1000)
    event_redshifts = sample_from_func(100 * n_events, merger_rate, full_z, *md_theta)
    inference = DrawnGWMergerRatePriorInference(H0_min, H0_max, theta_min, theta_max, fiducial_H0=fiducial_H0)
    cosmology = inference.fiducial_cosmology
    events = EventGenerator(fiducial_H0).from_redshifts(cosmology, event_redshifts, sigma_dl=0.1)
    if verbose:
        print(f"{len(events)} events were generated")

    initial = fiducial + np.random.randn(32, len(fiducial))
    nwalkers, ndim = initial.shape

    sampler = EnsembleSampler(nwalkers, ndim, inference.log_posterior, args=[events, full_z])
    if verbose:
        print("Starting MCMC")
    sampler.run_mcmc(initial, 30, progress=True)

    thin = round(np.max(sampler.get_autocorr_time()) / 2)
    samples = sampler.get_chain(discard=10, thin=2, flat=True)
    fig = corner(samples, labels=[r"$H_0$", r"$\alpha$", r"$\beta$", r"$c$"], truths=fiducial)
