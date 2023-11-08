import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from .utils import flat_cosmology, gaussian


def combine_posteriors(posterior_matrix, param_arr):
    """
    Return the normalised combined posterior given a posterior matrix
    and a parameter array as the domain of integration.
    """
    combined_posterior = np.ones_like(param_arr)
    n_events, n_param_samples = posterior_matrix.shape
    assert n_param_samples == len(param_arr)
    for i in range(n_events):
        # Normalize posterior for event i
        posterior_matrix[i, :] /= simpson(posterior_matrix[i, :], param_arr)

        # Normalize combined posterior with event i
        combined_posterior *= posterior_matrix[i, :]
        combined_posterior /= simpson(combined_posterior, param_arr)

    return combined_posterior, posterior_matrix


class HierarchicalBayesianLikelihood:
    def __init__(self) -> None:
        pass


class SimplifiedLikelihood(HierarchicalBayesianLikelihood):
    """
    Simplified likelihood based on arxiv:2212.08694.
    """

    def __init__(
        self,
        *args,
        sigma_constant=0.1,
        fiducial_H0=70,
        z_draw_max=1.4,
        dl_th=1550,
        ignore_z_error=False,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sigma_constant = sigma_constant
        self.fiducial_cosmology = flat_cosmology(fiducial_H0)
        self.dl_th = dl_th
        self.z_draw_max = z_draw_max
        self.ignore_z_error = ignore_z_error

    def uniform_p_rate(self, z_gal):
        p_rate = np.zeros_like(z_gal)
        p_rate[z_gal <= self.z_draw_max] = 1
        return p_rate / np.sum(p_rate)

    def draw_gw_events(self, z_gal, n_gw):
        if not self.ignore_z_error:
            raise NotImplementedError

        # Uniform merger probability on 0 < z < z_draw_max
        p_rate = self.uniform_p_rate(z_gal)

        # Get the "true" gw redshifts
        drawn_gw_zs = np.random.choice(z_gal, n_gw, p=p_rate)

        # Convert them into "true" gw luminosity distances using a fiducial cosmology
        drawn_gw_dls = (
            self.fiducial_cosmology.luminosity_distance(drawn_gw_zs).to("Mpc").value
        )

        # Convert true gw luminosity distances into measured values
        # drawn from a normal distribution consistent with the GW likelihood
        observed_gw_dls = np.array(
            [
                np.random.normal(gw_dl, gw_dl * self.sigma_constant)
                for gw_dl in drawn_gw_dls
            ]
        )
        return observed_gw_dls

    def population_prior(self, H0, z, z_gal):
        redshift_likelihood = self.redshift_likelihood(z, z_gal)
        redshift_prior = flat_cosmology(H0).differential_comoving_volume(z).value
        normalization = simpson(z, redshift_likelihood * redshift_prior)
        return redshift_likelihood * redshift_prior / normalization

    def redshift_likelihood(self, z, z_gal):
        """
        Compute galaxy redshift likelihood p(z_gal_measured | z_gal_true)
        See Eq. (17)
        """
        sigma = 0.0013 * (1 + z) ** 3
        return gaussian(z, z_gal, sigma)

    def gw_likelihood(self, dl, true_dl):
        """
        Compute gw likelihood L(dl_gw | z, H0)
        See Eq. (21)
        """
        sigma = self.sigma_constant * dl
        return gaussian(dl, true_dl, sigma)

    def detection_probability(self, dl):
        """
        Compute GW likelihood selection effects
        See Eq. (22)
        """
        sigma = self.sigma_constant * dl
        return 0.5 + 0.5 * erf((dl - self.dl_th) / np.sqrt(2) / sigma)

    def single_event_likelihood(self, gw_dl, H0_array, z_gal):
        """
        Compute single event likelihood
        Uses Eq. (15) when uncertainties on galaxy redshift are neglected
        """
        if not self.ignore_z_error:
            raise NotImplementedError

        likelihood_array = np.zeros_like(H0_array)

        p_rate = self.uniform_p_rate(z_gal)

        # Convert galaxy redshifts into luminosity distance using fixed H0
        fiducial_dl = self.fiducial_cosmology.luminosity_distance(z_gal).to("Mpc").value

        # Exploit that luminosity distance is linear with 1/H0
        # to efficiently compute dl for arbitrary H0
        fiducial_dl_times_H0 = fiducial_dl * self.fiducial_cosmology.H0.value
        for i, H0 in enumerate(H0_array):
            dl = fiducial_dl_times_H0 / H0
            numerator = np.sum(self.gw_likelihood(gw_dl, dl) * p_rate)
            denominator = np.sum(self.detection_probability(dl) * p_rate)
            likelihood_array[i] = numerator / denominator

        return likelihood_array

    def likelihood(self, gw_dl_array, H0_array, z_gal):
        if not self.ignore_z_error:
            raise NotImplementedError

        (n_gw,) = gw_dl_array.shape
        (n_H0,) = H0_array.shape
        likelihood_matrix = np.zeros((n_gw, n_H0))
        for i, gw_dl in enumerate(gw_dl_array):
            likelihood_matrix[i, :] = self.single_event_likelihood(
                gw_dl, H0_array, z_gal
            )

        return combine_posteriors(likelihood_matrix, H0_array)
