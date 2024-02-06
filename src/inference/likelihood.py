import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from .prior import UniformPrior
from .utils import (
    flat_cosmology,
    gaussian,
    lognormal,
    merger_rate,
    low_redshift_merger_rate,
    normalize,
    luminosity_distance,
)

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
        # Normalize combined posterior with event i
        combined_posterior *= normalize(posterior_matrix[i, :], param_arr)
        combined_posterior = normalize(combined_posterior, param_arr)

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
        sigma_dl=0.1,
        gw_likelihood_dist="normal",
        fiducial_H0=70,
        z_draw_max=1.4,
        dl_th=1550,
        max_redshift_err=0.015,
    ) -> None:
        self.sigma_dl = sigma_dl
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
        return luminosity_distance(self.fiducial_cosmology, z)

    def redshift_sigma(self, z):
        return np.minimum(0.033 * (1 + z), self.max_redshift_err)

    def redshift_likelihood(self, z, z_gal):
        """
        Compute galaxy redshift likelihood p(z_gal | z)

        See Eq. (17)
        """
        return gaussian(z, z_gal, self.redshift_sigma(z))

    def gw_likelihood(self, dl, true_dl):
        """
        Compute single-event gw likelihood L(dl_gw | z, H0)

        See Eq. (21)
        """
        is_normal = self.gw_likelihood_dist == "normal"
        sigma = self.sigma_dl * true_dl if is_normal else self.sigma_dl
        dist = gaussian if is_normal else lognormal
        return dist(dl, true_dl, sigma)

    def detection_probability(self, dl):
        """
        Return GW likelihood selection effects for a particular d_L

        See Eq. (22)
        """
        sigma = self.sigma_dl * dl
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

    def weighted_p_rate(self, z_gal, weights):
        p_rate = np.copy(weights)
        p_rate[z_gal > self.z_draw_max] = 0
        return p_rate / np.sum(p_rate)

    def p_rate(self, z_gal, weights=None):
        return self.weighted_p_rate(z_gal, weights) if weights is not None else self.uniform_p_rate(z_gal)


class DrawnGWCatalogSpeczInference(DrawnGWInference):
    """
    Implements likelihood model in the limit of perfect galaxy redshift measurements

    See Eq. (15)
    """

    def selection_effects(self, H0_array, z_gal, weights=None):
        """
        Return an array with GW likelihood selection effects for each H0 in the array
        """
        p_rate = self.p_rate(z_gal, weights)
        dl_by_H0_by_gal_matrix = self.dl_from_H0_array_and_z(H0_array, z_gal)
        detection_prob = self.detection_probability(dl_by_H0_by_gal_matrix)
        return np.dot(detection_prob, p_rate)

    def likelihood(self, gw_dl_array, H0_array, z_gal, n_dir=1, weights=None):
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
        for i, gws_in_dir_i in enumerate(gw_dl_array):
            z_gal_i = z_gal[i]
            weights_i = weights[i] if weights is not None else None
            p_rate = self.p_rate(z_gal_i, weights_i)
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


class DrawnGWCatalogPhotozInference(DrawnGWInference):
    """
    Implements full likelihood model

    See Eq. (29)
    """

    def p_cbc(self, z, z_gal, weights=None):
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

        if weights is not None:
            p_cbc *= weights.reshape(-1, 1)

        # Normalize each z_gal entry
        norm = simpson(p_cbc, z)
        # Discard zero norm galaxies: this happens because their redshifts falls outside [0, z_max] @ high sigma
        galaxies_in_range = np.nonzero(norm > 1e-6)
        p_cbc = p_cbc[galaxies_in_range] / norm[galaxies_in_range].reshape(-1, 1)

        # Sum likelihood * prior on z for over galaxies
        p_cbc = np.dot(weights[galaxies_in_range], p_cbc) if weights is not None else np.sum(p_cbc, axis=0)
        return normalize(p_cbc, z)

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

    def likelihood(self, gw_dl_array, H0_array, z, z_gal, n_dir=1, weights=None):
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
                weights_i = weights[i] if weights is not None else None
                p_rate = self.p_cbc(z, z_gal_i, weights_i)
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
            p_rate = self.p_cbc(z, z_gal, weights)
            selection_effects = self.selection_effects(H0_array, z, p_rate)
            for i, gw_dl in enumerate(gw_dl_array):
                likelihood_matrix[i, :] = self.gw_likelihood_array(gw_dl, H0_array, z, p_rate) / selection_effects

        return combine_posteriors(likelihood_matrix, H0_array)


class DrawnGWMergerRatePriorInference(DrawnGWInference):
    """
    Class implementing likelihood model using prior knowledge on the BBH merger rate per comoving volume.

    See arxiv:2103.14038
    """

    def __init__(self, z, prior: UniformPrior, *args, low_redshift=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.merger_rate = low_redshift_merger_rate if low_redshift else merger_rate

        # Prior limits
        expected_prior_dim = 2 if low_redshift else 4
        self.prior = prior
        if self.prior.ndim != expected_prior_dim:
            raise ValueError("Prior has incorrect dimensionality")

        # Pre-computed quantities to speed up likelihood computation
        self.dvc_dz_over_1pz = 4 * np.pi * self.fiducial_cosmology.differential_comoving_volume(z).value / (1 + z)
        self.fiducial_dl_times_fiducial_H0 = self.luminosity_distance(z) * self.H0

    def p_cbc(self, z, theta):
        """
        Return the merger rate prior on redshift, p(z|H0, alpha, beta, c)

        Uses Madau-Dickinson-like binary formation rate
        See https://arxiv.org/abs/2003.12152
        """
        # H0 dependence will cancel out in normalization
        sfr = self.merger_rate(z, *theta)
        # 1 + z factor in denominator accounts for the transformation from
        # detector frame time to source frame time
        p_cbc = self.dvc_dz_over_1pz * sfr
        return normalize(p_cbc, z)

    def selection_effects(self, true_dl, p_rate, z):
        """
        Return the estimated GW likelihood selection effects
        """
        detection_prob = self.detection_probability(true_dl)
        return simpson(detection_prob * p_rate, z)

    def likelihood(self, params, gw_dl_array, z):
        """
        Compute multi-event likelihood.

        Used for diagnostic purposes
        """
        H0, *theta = params
        p_rate = self.p_cbc(z, theta)
        # dl contains a factor of 1/H0
        true_dl = self.fiducial_dl_times_fiducial_H0 / H0
        # This is a n_events x n_z array
        numerator_over_z = np.array([self.gw_likelihood(gw_dl, true_dl) for gw_dl in gw_dl_array])
        numerator = simpson(numerator_over_z * p_rate, z)
        selection_effects = self.selection_effects(true_dl, p_rate, z)
        return np.prod(numerator / selection_effects)

    def log_likelihood(self, params, gw_dl_array, z):
        """
        Compute multi-event log-likelihood

        params = {H_0, \alpha, \beta, c}
        """
        H0, *theta = params
        p_rate = self.p_cbc(z, theta)
        # dl contains a factor of 1/H0
        true_dl = self.fiducial_dl_times_fiducial_H0 / H0
        # This is a n_events x n_z array
        numerator_over_z = np.array([self.gw_likelihood(gw_dl, true_dl) for gw_dl in gw_dl_array])
        numerator = simpson(numerator_over_z * p_rate, z)
        selection_effects = self.selection_effects(true_dl, p_rate, z)
        log_like = np.log(numerator) - np.log(selection_effects)
        return np.sum(log_like)

    def log_posterior(self, params, gw_dl_array, z):
        """
        Compute log-posterior
        """
        lp = self.prior.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        # Neglect uniform prior contribution
        return self.log_likelihood(params, gw_dl_array, z)
