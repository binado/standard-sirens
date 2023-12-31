import numpy as np
from scipy.special import erf, logsumexp
from scipy.integrate import simpson
from .utils import flat_cosmology, gaussian, lognormal
from .constants import eps

# logger = get_logger()

GW_LIKELIHOOD_DIST_OPTIONS = ("normal", "lognormal")


def combine_posteriors(posterior_matrix, param_arr):
    """
    Return the normalised combined posterior given a posterior matrix
    and a parameter array as the domain of integration.
    """
    # # Normalize posterior matrix, add constant eps to avoid underflow
    # norm = simpson(posterior_matrix, param_arr) + eps
    # posterior_matrix /= norm.reshape(-1, 1)
    # # Compute posterior with exp(logsumexp) trick to prevent numerical underflows
    # log_posterior_matrix = np.log(posterior_matrix)
    # combined_posterior = np.exp(logsumexp(log_posterior_matrix, axis=0))
    # combined_norm = simpson(combined_posterior, param_arr)

    combined_posterior = np.ones_like(param_arr)
    n_events, n_param_samples = posterior_matrix.shape
    assert n_param_samples == len(param_arr)
    for i in range(n_events):
        # Normalize posterior for event i
        norm = simpson(posterior_matrix[i, :], param_arr) + eps
        posterior_matrix[i, :] /= norm

        # Normalize combined posterior with event i
        combined_posterior *= posterior_matrix[i, :]
        combined_norm = simpson(combined_posterior, param_arr) + eps
        combined_posterior /= combined_norm

    # combined_posterior /= simpson(combined_posterior, param_arr)
    # if combined_norm <= 0:
    #     print("Null posterior encountered")
    # else:
    #     combined_posterior /= combined_norm
    return combined_posterior, posterior_matrix


class HierarchicalBayesianInference:
    def __init__(self) -> None:
        pass


class DrawnGWInference(HierarchicalBayesianInference):
    """
    Base class with helper methods for implementing
    the likelihood models based on arxiv:2212.08694 and arxiv:2103.14038.
    """

    def __init__(
        self,
        sigma_constant=0.1,
        gw_likelihood_dist="normal",
        fiducial_H0=70,
        z_draw_max=1.4,
        dl_th=1550,
        max_redshift_err=0.015,
    ) -> None:
        self.sigma_constant = sigma_constant
        self.fiducial_cosmology = flat_cosmology(fiducial_H0)
        self.dl_th = dl_th
        self.z_draw_max = z_draw_max
        self.max_redshift_err = max_redshift_err

        if gw_likelihood_dist not in GW_LIKELIHOOD_DIST_OPTIONS:
            raise ValueError('Parameter should either be "normal" or "lognormal".')

        self.gw_likelihood_dist = gw_likelihood_dist

    @property
    def H0(self):
        return self.fiducial_cosmology.H0.value

    def luminosity_distance(self, z):
        """
        Return value of luminosity distance at redshift z in Mpc
        for a flat LambdaCDM cosmology with fiducial H0
        """
        return self.fiducial_cosmology.luminosity_distance(z).to("Mpc").value

    def redshift_sigma(self, z):
        return np.minimum(0.013 * (1 + z) ** 3, self.max_redshift_err)

    def redshift_likelihood(self, z, z_gal):
        """
        Compute galaxy redshift likelihood p(z_gal | z)

        See Eq. (17)
        """
        return gaussian(z, z_gal, self.redshift_sigma(z))

    def draw_from_redshift_likelihood(self, z_true):
        return z_true + self.redshift_sigma(z_true) * np.random.standard_normal(len(z_true))

    def gw_likelihood(self, dl, true_dl):
        """
        Compute gw likelihood L(dl_gw | z, H0)

        See Eq. (21)
        """
        is_normal = self.gw_likelihood_dist == "normal"
        sigma = self.sigma_constant * true_dl if is_normal else self.sigma_constant
        dist = gaussian if is_normal else lognormal
        return dist(dl, true_dl, sigma)

    def detection_probability(self, dl):
        """
        Return GW likelihood selection effects for a particular d_L

        See Eq. (22)
        """
        sigma = self.sigma_constant * dl
        x = (self.dl_th - dl) / sigma
        return 0.5 * (1.0 + erf(x / np.sqrt(2)))

    def dl_from_H0_array_and_z(self, H0_array, z):
        """
        Return a n_H0 x n_z grid of 'true' luminosity distance values
        """
        # Convert redshifts into luminosity distance using fixed H0
        fiducial_dl = self.luminosity_distance(z)
        # Exploit that luminosity distance * H0 is independent of H0 for fixed fixed \Omega_i's
        # to efficiently compute dl for arbitrary H0
        fiducial_dl_times_H0 = fiducial_dl * self.H0
        return np.array([fiducial_dl_times_H0 / H0 for H0 in H0_array])

    def uniform_p_rate(self, z_gal):
        p_rate = np.zeros_like(z_gal)
        p_rate[z_gal <= self.z_draw_max] = 1
        return p_rate / np.sum(p_rate)

    def draw_gw_events(self, z_gal, sigma_constant, n_gw=None, gw_noise=None):
        # Uniform merger probability on 0 < z < z_draw_max
        p_rate = self.uniform_p_rate(z_gal)

        # Get the "true" gw redshifts
        drawn_gw_zs = np.random.choice(z_gal, n_gw, p=p_rate)

        # Convert them into "true" gw luminosity distances using a fiducial cosmology
        drawn_gw_dls = self.luminosity_distance(drawn_gw_zs)

        # Compute noise
        assert n_gw is not None or gw_noise is not None, "Either n_gw or gw_noise_arr must be set"
        noise = gw_noise if gw_noise is not None else np.random.standard_normal(n_gw)

        # Convert true gw luminosity distances into measured values
        # drawn from a normal distribution consistent with the GW likelihood
        sigma_dl = drawn_gw_dls * sigma_constant
        observed_gw_dls = noise * sigma_dl + drawn_gw_dls
        # Filter events whose dL exceeds threshold
        return observed_gw_dls[observed_gw_dls < self.dl_th]


class DrawnGWCatalogPerfectRedshiftInference(DrawnGWInference):
    """
    Implements likelihood model in the limit of perfect galaxy redshift measurements

    See Eq. (15)
    """

    def selection_effects(self, H0_array, z_gal):
        """
        Return an array with GW likelihood selection effects for each H0 in the array
        """
        p_rate = self.uniform_p_rate(z_gal)
        dl_by_H0_by_gal_matrix = self.dl_from_H0_array_and_z(H0_array, z_gal)
        detection_prob = self.detection_probability(dl_by_H0_by_gal_matrix)
        return np.dot(detection_prob, p_rate)

    def likelihood(self, gw_dl_array, H0_array, z_gal, n_dir=1):
        n_H0 = H0_array.shape[0]
        n_gw = np.sum([len(gw_dl) for gw_dl in gw_dl_array])
        current_gw_idx = 0
        likelihood_matrix = np.ones((n_gw, n_H0))

        # Each direction has a different set of candidate galaxies
        # If n_dir = 1, put single direction in list to iterate over
        # as a general case
        gw_dl_array = gw_dl_array if n_dir > 1 else [gw_dl_array]
        z_gal = z_gal if n_dir > 1 else [z_gal]

        assert len(z_gal) == len(gw_dl_array)
        assert len(z_gal) == n_dir
        # Loop through directions
        for gws_in_dir_i, z_gal_i in zip(gw_dl_array, z_gal):
            p_rate = self.uniform_p_rate(z_gal_i)
            dl_by_H0_by_gal_matrix = self.dl_from_H0_array_and_z(H0_array, z_gal_i)
            selection_effects = np.dot(self.detection_probability(dl_by_H0_by_gal_matrix), p_rate)
            # Loop through GWs for a fixed direction
            for gw_dl_j in gws_in_dir_i:
                # sum over galaxies of p(d_L | d_L(H0, z_gal)) for each H0
                numerator = np.dot(self.gw_likelihood(gw_dl_j, dl_by_H0_by_gal_matrix), p_rate)
                likelihood_matrix[current_gw_idx, :] = numerator / selection_effects
                current_gw_idx += 1
        assert current_gw_idx == n_gw

        return combine_posteriors(likelihood_matrix, H0_array)


class DrawnGWCatalogFullInference(DrawnGWInference):
    """
    Implements full likelihood model

    See Eq. (29)
    """

    def p_cbc(self, z, z_gal):
        """
        Return \Sum_i p(z_gal_i | z) p_bg (z) p_rate (z)

        See Eqs. (16), (29)
        """
        p_rate = 1.0 / (1.0 + z)  # See Eq. (5)
        # Use fiducial H0, as dependence cancels out in normalization
        p_bg = self.fiducial_cosmology.differential_comoving_volume(z).value

        # Create n_z x n_z_gal redshift likelihood matrix
        z_gal_by_z_likelihood = np.array([self.redshift_likelihood(z, z_gal_i) for z_gal_i in z_gal])

        # This is an n_gal x n_z matrix
        p_cbc = z_gal_by_z_likelihood * p_bg * p_rate

        # Normalize each z_gal entry
        norm = simpson(p_cbc, z)
        # Discard zero norm galaxies: this happens because their redshifts falls outside [0, z_max] @ high sigma
        galaxies_in_range = np.nonzero(norm > 1e-6)
        p_cbc = p_cbc[galaxies_in_range] / norm[galaxies_in_range].reshape(-1, 1)

        # Sum likelihood * prior on z for over galaxies
        p_cbc = np.sum(p_cbc, axis=0)
        return p_cbc / simpson(p_cbc, z)

    def gw_likelihood_array(self, dl, H0_array, z, p_rate):
        """
        Return the sum over galaxies of p(d_L | d_L(H0, z_gal)) for each H0
        """
        dl_by_H0_by_z_matrix = self.dl_from_H0_array_and_z(H0_array, z)
        integrand = self.gw_likelihood(dl, dl_by_H0_by_z_matrix) * p_rate
        return simpson(integrand, z)

    def selection_effects(self, H0_array, z, p_rate):
        """
        Return an array with GW likelihood selection effects for each H0 in the array
        """
        dl_by_H0_by_z_matrix = self.dl_from_H0_array_and_z(H0_array, z)
        detection_prob = self.detection_probability(dl_by_H0_by_z_matrix)
        return simpson(detection_prob * p_rate, z)

    def likelihood(self, gw_dl_array, H0_array, z, z_gal, n_dir=1):
        n_H0 = H0_array.shape[0]
        n_gw = np.sum([len(gw_dl) for gw_dl in gw_dl_array])
        current_gw_idx = 0
        likelihood_matrix = np.ones((n_gw, n_H0))

        # Each direction has a different set of candidate galaxies
        if n_dir > 1:
            assert len(z_gal) == len(gw_dl_array)
            assert len(z_gal) == n_dir
            # Loop through directions
            for i, (gws_in_dir_i, z_gal_i) in enumerate(zip(gw_dl_array, z_gal)):
                p_rate = self.p_cbc(z, z_gal_i)
                selection_effects = self.selection_effects(H0_array, z, p_rate)
                # Loop through GWs for a fixed direction
                for gw_dl_j in gws_in_dir_i:
                    likelihood_matrix[current_gw_idx, :] = (
                        self.gw_likelihood_array(gw_dl_j, H0_array, z, p_rate) / selection_effects
                    )
                    current_gw_idx += 1
            assert current_gw_idx == n_gw
        # GWs share the same set of host galaxies
        else:
            p_rate = self.p_cbc(z, z_gal)
            selection_effects = self.selection_effects(H0_array, z, p_rate)
            for i, gw_dl in enumerate(gw_dl_array):
                likelihood_matrix[i, :] = self.gw_likelihood_array(gw_dl, H0_array, z, p_rate) / selection_effects

        return combine_posteriors(likelihood_matrix, H0_array)


class DrawnGWMergerRatePriorInference(DrawnGWInference):
    """
    Class implementing likelihood model using prior knowledge on the
    star formation rate.

    See arxiv:2103.14038
    """

    def p_cbc(self, z):
        """
        Return the probability of a CBC at z, p_pop(z | H_0)

        Uses Madau-Dickinson SFR (arxiv:1403.0007)
        """
        # H0 dependence will cancel out in normalization
        dvc_dz = self.fiducial_cosmology.differential_comoving_volume(z).value
        sfr = np.power(1 + z, 2.7) / (1.0 + np.power((1 + z) / 2.9, 5.6))
        # 1 + z factor in denominator accounts for the transformation from
        # detector frame time to source frame time
        p_cbc = dvc_dz * sfr / (1 + z)
        return p_cbc / simpson(p_cbc, z)

    def gw_likelihood_array(self, dl, H0_array, z):
        """
        Return the sum over galaxies of p(d_L | d_L(H0, z)) for each H0
        """
        p_rate = self.p_cbc(z)
        dl_by_H0_by_z_matrix = self.dl_from_H0_array_and_z(H0_array, z)
        integrand = self.gw_likelihood(dl, dl_by_H0_by_z_matrix) * p_rate
        return simpson(integrand, z)

    def selection_effects(self, H0_array, z):
        """
        Return an array with GW likelihood selection effects for each H0 in the array
        """
        p_rate = self.p_cbc(z)
        dl_by_H0_by_z_matrix = self.dl_from_H0_array_and_z(H0_array, z)
        detection_prob = self.detection_probability(dl_by_H0_by_z_matrix)
        return simpson(detection_prob * p_rate, z)

    def likelihood(self, gw_dl_array, H0_array, z):
        gw_dl_flatenned_array = np.concatenate(gw_dl_array)
        likelihood_matrix = np.array([self.gw_likelihood_array(gw_dl, H0_array, z) for gw_dl in gw_dl_flatenned_array])
        selection_effects = self.selection_effects(H0_array, z)
        likelihood_matrix /= selection_effects
        return combine_posteriors(likelihood_matrix, H0_array)
