import pickle
import numpy as np
from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from dynesty import DynamicNestedSampler
from dynesty.results import Results

from .prior import UniformPrior


class MCMCStrategy:
    """
    A wrapper class over `emcee`'s `EnsembleSampler`.

    See https://emcee.readthedocs.io/en/stable/user/sampler/
    """

    def __init__(self, nwalkers, ndim, logprob, filename=None, reset=False, **kwargs) -> None:
        self.nwalkers, self.ndim = nwalkers, ndim
        self.backend = None
        if filename:
            self.backend = HDFBackend(filename)
            if reset:
                self.backend.reset(nwalkers, ndim)

        self.sampler = EnsembleSampler(nwalkers, ndim, logprob, backend=self.backend, **kwargs)

    def run(self, *args, **kwargs):
        self.sampler.run_mcmc(*args, **kwargs)

    def initial_around(self, a, var, prior: UniformPrior):
        if len(a) != self.ndim or len(var) != self.ndim or self.ndim != self.ndim:
            raise ValueError("'a', 'var' and 'prior' must have consistent dimensions with sampler")

        initial = a + var * np.random.standard_normal((self.nwalkers, self.ndim))
        is_finite = np.all(np.isfinite([prior.log_prior(walker) for walker in initial]))
        if not is_finite:
            raise ValueError("Initial state not within allowed prior range")

        return initial

    @staticmethod
    def read_from_file(filename):
        return HDFBackend(filename, read_only=True)


class NestedStrategy:
    """
    A wrapper class over `dynesty`'s top-level interface.

    See https://dynesty.readthedocs.io/en/stable/api.html#module-dynesty.dynesty
    """

    def __init__(self, loglike, prior: UniformPrior, filename=None, **kwargs) -> None:
        self.prior = prior
        self.ndim = prior.ndim
        self.filename = filename
        self.sampler = DynamicNestedSampler(loglike, prior.prior_transform, self.ndim, **kwargs)
        self.results = None

    def run(self, **kwargs):
        self.sampler.run_nested(**kwargs)
        self.results = self.sampler.results
        if self.filename:
            self.save_to_file(self.filename)

    def save_to_file(self, filename):
        with open(filename, "r+b") as f:
            pickle.dump(self.results.asdict(), f)

    @staticmethod
    def read_from_file(filename):
        with open(filename, "rb") as f:
            results_as_dict = pickle.load(f)
            return Results(results_as_dict)
