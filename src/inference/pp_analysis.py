import math
from .logging_utils import get_logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import binom
import h5py
from .likelihood import (
    DrawnGWCatalogPerfectRedshiftInference,
    DrawnGWCatalogFullInference,
)


def plot_sigma(ax, n, arr, color="tab:gray"):
    for ci, alpha in zip([0.68, 0.95, 0.997], [0.1, 0.15, 0.2]):
        edge_of_bound = (1.0 - ci) / 2.0
        lower = binom.ppf(1 - edge_of_bound, n, arr) / n
        upper = binom.ppf(edge_of_bound, n, arr) / n
        lower[0] = 0
        upper[0] = 0
        ax.plot(arr, lower, c=color)
        ax.plot(arr, upper, c=color)
        ax.fill_between(arr, lower, upper, alpha=alpha, color="k")


class PPAnalysis:
    def __init__(self, n_sim=100, logger=None) -> None:
        self.n_sim = n_sim
        self.ci_array = np.empty(n_sim)
        self.logger = logger

    def draw_parameters(self, param_array: np.ndarray):
        param_min, param_max = param_array[0], param_array[-1]
        drawn_true_params = np.random.uniform(param_min, param_max, self.n_sim)
        drawn_true_params_indexes = np.searchsorted(param_array, drawn_true_params)
        return drawn_true_params, drawn_true_params_indexes

    def make_histplot(self):
        assert self.ci_array is not None, "run_inference method must be called first"
        assert self.n_sim > 0

        counts, bins, _ = plt.hist(self.ci_array, bins=self.n_sim, density=True, cumulative=True)
        return counts, bins

    def plot(self, ax, counts, bins, **kwargs):
        bins_midpoints = 0.5 * (bins[1:] + bins[:-1])
        assert len(bins_midpoints) == len(counts)
        ax.plot(bins_midpoints, counts, **kwargs)
        ax.set_xlabel("Credible interval (CI)")
        ax.set_ylabel("Fraction of runs in CI")
        ax.grid()

    def save_data(self, filename, dataset):
        with h5py.File(filename, "a") as f:
            try:
                del f[dataset]
            except KeyError:
                pass
            f.create_dataset(dataset, data=self.ci_array)


class PPAnalysisPerfectRedshift(PPAnalysis):
    def run_inference(self, H0_array, z_gal, n_gw, n_dir, c):
        true_H0s, true_H0_indexes = self.draw_parameters(H0_array)
        n_gw_per_dir = math.ceil(n_gw / n_dir)

        for i, H0 in enumerate(true_H0s):
            inference = DrawnGWCatalogPerfectRedshiftInference(fiducial_H0=H0, sigma_constant=c)
            events = [inference.draw_gw_events(z_gal_i, c, n_gw_per_dir) for z_gal_i in z_gal]
            posterior, _ = inference.likelihood(events, H0_array, z_gal, n_dir=n_dir)
            cdf = cumulative_trapezoid(posterior, H0_array)
            assert np.all(cdf >= 0), "Negative cdf"
            H0_index = min(true_H0_indexes[i], len(cdf) - 1)
            # Compute symmetric CI
            self.ci_array[i] = 1.0 - 2 * min(cdf[H0_index], 1 - cdf[H0_index])


class PPAnalysisFullInference(PPAnalysis):
    def run_inference(self, H0_array, z, z_gal, n_gw, n_dir, c):
        true_H0s, true_H0_indexes = self.draw_parameters(H0_array)
        n_gw_per_dir = math.ceil(n_gw / n_dir)

        for i, H0 in enumerate(true_H0s):
            inference = DrawnGWCatalogFullInference(fiducial_H0=H0, sigma_constant=c)
            events = [inference.draw_gw_events(z_gal_i, c, n_gw_per_dir) for z_gal_i in z_gal]
            posterior, _ = inference.likelihood(events, H0_array, z, z_gal, n_dir=n_dir)
            cdf = cumulative_trapezoid(posterior, H0_array)
            assert np.all(cdf >= 0), "Negative cdf"
            H0_index = min(true_H0_indexes[i], len(cdf) - 1)
            # Compute symmetric CI
            self.ci_array[i] = 1.0 - 2 * min(cdf[H0_index], 1 - cdf[H0_index])


def pp_analysis_perfect_redshift(H0_array, N, z_gal, n_gw, n_dir, c):
    H0_min, H0_max = H0_array[0], H0_array[-1]
    true_H0s = np.random.uniform(H0_min, H0_max, N)
    true_H0_indexes = np.searchsorted(H0_array, true_H0s)
    n_gw_per_dir = math.ceil(n_gw / n_dir)
    ci_arr = np.empty(N)

    for i, H0 in enumerate(true_H0s):
        inference = DrawnGWCatalogPerfectRedshiftInference(fiducial_H0=H0, sigma_constant=c)
        events = [inference.draw_gw_events(z_gal_i, n_gw_per_dir, c) for z_gal_i in z_gal]
        posterior, _ = inference.likelihood(events, H0_array, z_gal, n_dir=n_dir)
        cdf = cumulative_trapezoid(posterior, H0_array)
        H0_index = true_H0_indexes[i]
        # Compute symmetric CI
        ci_arr[i] = 1.0 - 2 * min(cdf[H0_index], 1 - cdf[H0_index])

    logger.info("ci array: %s", ci_arr)
    counts, bins, _ = plt.hist(ci_arr, bins=N, density=True, cumulative=True)
    return counts, 0.5 * (bins[1:] + bins[:-1])


def pp_analysis_full_inference(H0_array, N, z, z_gal, n_gw, n_dir, c):
    H0_min, H0_max = H0_array[0], H0_array[-1]
    true_H0s = np.random.uniform(H0_min, H0_max, N)
    true_H0_indexes = np.searchsorted(H0_array, true_H0s)
    n_dir = len(z_gal)
    n_gw_per_dir = math.ceil(n_gw / n_dir)
    ci_arr = np.empty(N)

    for i, H0 in enumerate(true_H0s):
        inference = DrawnGWCatalogFullInference(fiducial_H0=H0, sigma_constant=c)
        events = [inference.draw_gw_events(z_gal_i, n_gw_per_dir, c) for z_gal_i in z_gal]
        posterior, _ = inference.likelihood(events, H0_array, z, z_gal, n_dir=n_dir)
        cdf = cumulative_trapezoid(posterior, H0_array)
        H0_index = true_H0_indexes[i]
        # Compute symmetric CI
        ci_arr[i] = 1.0 - 2 * min(cdf[H0_index], 1 - cdf[H0_index])

    logger.info("ci array: %s", ci_arr)
    counts, bins, _ = plt.hist(ci_arr, bins=N, density=True, cumulative=True)
    logger.info("bins: %s", bins)
    return counts, 0.5 * (bins[1:] + bins[:-1])
