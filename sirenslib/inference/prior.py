import numpy as np


class Parameters:

    def __init__(self, labels, plot_labels, truths=None, **fixed_params):
        self.labels = labels
        self.plot_labels = plot_labels
        self.truths = list(truths)
        self.ndim = len(self.labels)
        self.fixed_params = fixed_params

        # Keep track of free parameters (no default values)
        mask = [label not in fixed_params for label in self.labels]
        self.free_params_labels = [label for label, is_free in zip(labels, mask) if is_free]
        self.free_params_plot_labels = [label for label, is_free in zip(plot_labels, mask) if is_free]
        self.nfree_dim = len(self.free_params_labels)

    def __call__(self, free_params):
        param_dict = dict(zip(self.free_params_labels, free_params))
        param_dict.update(**self.fixed_params)
        return np.array([param_dict[label] for label in self.labels])

    def asdict(self):
        return {
            "params": self.labels,
            "free_params": self.free_params_labels,
            "fixed_params": self.fixed_params,
            "truths": self.truths,
            "plot_labels": self.plot_labels,
            "free_params_plot_labels": self.free_params_plot_labels,
        }


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

    def __str__(self):
        return f"Uniform({self.prior_min}, {self.prior_max})"

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
