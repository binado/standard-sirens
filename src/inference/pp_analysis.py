from tqdm import tqdm
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import binom
import h5py
from .utils import EventGenerator
from .likelihood import (
    DrawnGWCatalogSpeczInference,
    DrawnGWCatalogPhotozInference,
)


def plot_cumulative(ax, counts, bins, *args, **kwargs):
    cumulative_counts = np.cumsum(counts) / np.sum(counts)
    bins_midpoints = 0.5 * (bins[1:] + bins[:-1])
    assert len(bins_midpoints) == len(cumulative_counts)
    ax.plot(bins_midpoints, cumulative_counts, *args, **kwargs)


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


def draw_parameters(param_array: np.ndarray, n_sim: int):
    param_min, param_max = param_array[0], param_array[-1]
    drawn_true_params = np.random.uniform(param_min, param_max, n_sim)
    drawn_true_params_indexes = np.searchsorted(param_array, drawn_true_params)
    return drawn_true_params, drawn_true_params_indexes


def save_data(filename, dataset, data):
    with h5py.File(filename, "a") as f:
        try:
            del f[dataset]
        except KeyError:
            pass
        f.create_dataset(dataset, data=data)


def pp_analysis_specz(
    n_sim, H0_array, sigma_dl, z_gal, n_gw_per_dir, event_weights=None, inference_weights=None, desc=None
):
    true_H0s, true_H0_indexes = draw_parameters(H0_array, n_sim)
    generator = EventGenerator()
    n_dir = len(z_gal)
    ci_array = np.empty(n_sim)

    for i, H0 in tqdm(enumerate(true_H0s), total=n_sim, desc=desc):
        inference = DrawnGWCatalogSpeczInference(fiducial_H0=H0, sigma_dl=sigma_dl)
        events = generator.from_catalog(inference.fiducial_cosmology, z_gal, sigma_dl, n_gw_per_dir, event_weights)
        posterior, _ = inference.likelihood(events, H0_array, z_gal, n_dir=n_dir, weights=inference_weights)
        cdf = cumulative_trapezoid(posterior, H0_array)
        assert np.all(cdf >= 0), "Negative cdf"
        true_H0_index = min(true_H0_indexes[i], len(cdf) - 1)
        ci_array[i] = 1.0 - 2 * min(cdf[true_H0_index], 1 - cdf[true_H0_index])

    counts, bins = np.histogram(ci_array, bins=n_sim, density=True)
    return ci_array, counts, bins


def pp_analysis_photoz(
    n_sim, H0_array, sigma_dl, z, z_gal, n_gw_per_dir, event_weights=None, inference_weights=None, desc=None
):
    true_H0s, true_H0_indexes = draw_parameters(H0_array, n_sim)
    generator = EventGenerator()
    n_dir = len(z_gal)
    ci_array = np.empty(n_sim)

    for i, H0 in tqdm(enumerate(true_H0s), total=n_sim, desc=desc):
        inference = DrawnGWCatalogPhotozInference(fiducial_H0=H0, sigma_dl=sigma_dl)
        events = generator.from_catalog(inference.fiducial_cosmology, z_gal, sigma_dl, n_gw_per_dir, event_weights)
        posterior, _ = inference.likelihood(events, H0_array, z, z_gal, n_dir=n_dir, weights=inference_weights)
        cdf = cumulative_trapezoid(posterior, H0_array)
        assert np.all(cdf >= 0), "Negative cdf"
        true_H0_index = min(true_H0_indexes[i], len(cdf) - 1)
        ci_array[i] = 1.0 - 2 * min(cdf[true_H0_index], 1 - cdf[true_H0_index])

    counts, bins = np.histogram(ci_array, bins=n_sim, density=True)
    return ci_array, counts, bins
