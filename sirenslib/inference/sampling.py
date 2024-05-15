import abc
import pickle

import numpy as np
from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from dynesty import DynamicNestedSampler
from dynesty.results import Results
import json

SAMPLING_STRATEGIES = {
    "mcmc": {"name": "MCMC", "format": "hdf5"},
    "nested": {"name": "Nested Sampling", "format": "pickle"},
}
FORMATS = ["pickle", "hdf5"]


class SamplingStrategy(abc.ABC):
    name = None
    format = "pickle"

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass


class MCMCStrategy(SamplingStrategy):
    """
    A wrapper class over `emcee`'s `EnsembleSampler`.

    See https://emcee.readthedocs.io/en/stable/user/sampler/
    """

    name = SAMPLING_STRATEGIES["mcmc"]
    format = "hdf5"

    def __init__(self, nwalkers, ndim, logprob, filename=None, reset=False, **kwargs):
        super().__init__()
        self.nwalkers, self.ndim = nwalkers, ndim
        self.backend = None
        if filename:
            self.backend = HDFBackend(filename)
            if reset:
                self.backend.reset(nwalkers, ndim)

        self.sampler = EnsembleSampler(nwalkers, ndim, logprob, backend=self.backend, **kwargs)

    def run(self, *args, **kwargs):
        self.sampler.run_mcmc(*args, **kwargs)

    def initial_around(self, a, var, prior):
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


class NestedStrategy(SamplingStrategy):
    """
    A wrapper class over `dynesty`'s top-level interface.

    See https://dynesty.readthedocs.io/en/stable/api.html#module-dynesty.dynesty
    """

    name = SAMPLING_STRATEGIES["nested"]
    format = "pickle"

    def __init__(self, loglike, prior, filename=None, **kwargs):
        super().__init__()
        self.prior = prior
        self.ndim = prior.ndim
        self.filename = filename
        self.sampler = DynamicNestedSampler(loglike, prior.prior_transform, self.ndim, **kwargs)

    def run(self, *args, **kwargs):
        self.sampler.run_nested(**kwargs)
        results = self.sampler.results.asdict()
        if self.filename:
            with open(self.filename, "wb") as f:
                pickle.dump(results, f)

    @staticmethod
    def read_from_file(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            return Results(obj)


class SamplingRun:
    def __init__(self, params, prior, strategy):
        """Create a new SamplingRun instance.

        Parameters
        ----------
        params : Parameters
            The parameters of the model
        prior : UniformPrior
            The prior on the parameters
        strategy : SamplingStrategy
            The sampling strategy of choice
        """
        self.params = params
        self.prior = prior
        self.strategy = strategy

    def asdict(self, **attrs):
        val = dict(strategy=self.strategy.name, prior=str(self.prior))
        val.update(**self.params.asdict(), **attrs)
        return val

    def save_to_json(self, filename, **attrs):
        with open(filename, "w", encoding="utf8") as f:
            json.dump(self.asdict(**attrs), f)
