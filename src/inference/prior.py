import numpy as np


class UniformPrior:
    def __init__(self, prior_min, prior_max) -> None:
        if len(prior_max) != len(prior_min):
            raise ValueError("Prior bounds do not have the same dimension")

        self.prior_min = prior_min
        self.prior_max = prior_max
        self.ndim = len(prior_max)
        self.norm = np.prod(prior_max - prior_min)
        if self.norm < 0:
            raise ValueError("Prior has negative norm")

        self.log_norm = -np.log(self.norm)

    @property
    def interval(self):
        return self.prior_max - self.prior_min

    def log_prior(self, prior):
        if np.all(prior >= self.prior_min) and np.all(prior <= self.prior_max):
            return self.log_norm
        return -np.inf

    def prior_transform(self, u):
        average = 0.5 * (self.prior_max + self.prior_min)
        return self.interval * (u - 0.5) + average
