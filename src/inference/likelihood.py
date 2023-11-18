import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from .utils import flat_cosmology, gaussian, lognormal

GW_LIKELIHOOD_DIST_OPTIONS = ("normal", "lognormal")


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
        # combined_posterior /= simpson(combined_posterior, param_arr)

    combined_posterior /= simpson(combined_posterior, param_arr)
    return combined_posterior, posterior_matrix


class HierarchicalBayesianInference:
    def __init__(self) -> None:
        pass


class DrawnGWInference(HierarchicalBayesianInference):
    """
    Base class with helper methods for implementing
    the likelihood model based on arxiv:2212.08694.
    """

    def __init__(
        self,
        sigma_constant=0.1,
        gw_likelihood_dist="normal",
        fiducial_H0=70,
        z_draw_max=1.4,
        dl_th=1550,
        max_redshift_err=0.0015,
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

    def redshift_likelihood(self, z, z_gal):
        """
        Compute galaxy redshift likelihood p(z_gal_measured | z_gal_true)

        See Eq. (17)
        """
        sigma = np.minimum(0.0013 * (1 + z) ** 3, self.max_redshift_err)
        return gaussian(z, z_gal, sigma)

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

    def draw_gw_events(self, z_gal, n_gw):
        # Uniform merger probability on 0 < z < z_draw_max
        p_rate = self.uniform_p_rate(z_gal)

        # Get the "true" gw redshifts
        drawn_gw_zs = np.random.choice(z_gal, n_gw, p=p_rate)

        # Convert them into "true" gw luminosity distances using a fiducial cosmology
        drawn_gw_dls = self.luminosity_distance(drawn_gw_zs)

        # Convert true gw luminosity distances into measured values
        # drawn from a normal distribution consistent with the GW likelihood
        sigma_dl = drawn_gw_dls * self.sigma_constant
        observed_gw_dls = np.random.standard_normal(n_gw) * sigma_dl + drawn_gw_dls
        # Filter events whose dL exceeds threshold
        return observed_gw_dls[observed_gw_dls < self.dl_th]


class DrawnGWCatalogPerfectRedshiftInference(DrawnGWInference):
    """
    Implements likelihood model in the limit of perfect galaxy redshift measurements

    See Eq. (15)
    """

    def gw_likelihood_array(self, dl, H0_array, z_gal):
        """
        Return the sum over galaxies of p(d_L | d_L(H0, z_gal)) for each H0
        """
        p_rate = self.uniform_p_rate(z_gal)
        dl_by_H0_by_gal_matrix = self.dl_from_H0_array_and_z(H0_array, z_gal)
        return np.dot(self.gw_likelihood(dl, dl_by_H0_by_gal_matrix), p_rate)

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
        if n_dir > 1:
            assert len(z_gal) == len(gw_dl_array)
            assert len(z_gal) == n_dir
            # Loop through directions
            for i, (gws_in_dir_i, z_gal_i) in enumerate(zip(gw_dl_array, z_gal)):
                selection_effects = self.selection_effects(H0_array, z_gal_i)
                # Loop through GWs for a fixed direction
                for gw_dl_j in gws_in_dir_i:
                    likelihood_matrix[current_gw_idx, :] = (
                        self.gw_likelihood_array(gw_dl_j, H0_array, z_gal_i)
                        / selection_effects
                    )
                    current_gw_idx += 1
            assert current_gw_idx == n_gw
        # GWs share the same set of host galaxies
        else:
            selection_effects = self.selection_effects(H0_array, z_gal)
            for i, gw_dl in enumerate(gw_dl_array):
                likelihood_matrix[i, :] = (
                    self.gw_likelihood_array(gw_dl, H0_array, z_gal) / selection_effects
                )

        return combine_posteriors(likelihood_matrix, H0_array)


class DrawnGWFullLikelihood(DrawnGWLikelihood):
    """
    Implements full likelihood model

    See Eq. (29)
    """

    def p_cbc(self, z, z_gal, normalize=False):
        """
        Return \Sum_i p(z_gal_i | z) p_bg (z) p_rate (z)

        See Eqs. (16), (29)
        """
        p_rate = 1 / (1 + z)  # See Eq. (5)
        p_bg = self.fiducial_cosmology.differential_comoving_volume(z).value
        # Create n_z x n_z_gal redshift likelihood matrix
        z_gal_by_z_likelihood = np.array(
            [self.redshift_likelihood(z, z_gal_i) for z_gal_i in z_gal]
        )

        # Sum likelihood * prior on z for over galaxies
        p_cbc = np.sum(z_gal_by_z_likelihood * p_bg * p_rate, axis=0)
        if normalize:
            p_cbc /= simpson(p_cbc, z)

        return p_cbc

    def gw_likelihood_array(self, dl, H0_array, z, z_gal):
        """
        Return the sum over galaxies of p(d_L | d_L(H0, z_gal)) for each H0
        """
        p_rate = self.p_cbc(z, z_gal, normalize=True)
        dl_by_H0_by_z_matrix = self.dl_from_H0_array_and_z(H0_array, z)
        return np.dot(self.gw_likelihood(dl, dl_by_H0_by_z_matrix), p_rate)

    def selection_effects(self, H0_array, z, z_gal):
        """
        Return an array with GW likelihood selection effects for each H0 in the array
        """
        p_rate = self.p_cbc(z, z_gal, normalize=True)
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
                selection_effects = self.selection_effects(H0_array, z, z_gal_i)
                # Loop through GWs for a fixed direction
                for gw_dl_j in gws_in_dir_i:
                    likelihood_matrix[current_gw_idx, :] = (
                        self.gw_likelihood_array(gw_dl_j, H0_array, z, z_gal_i)
                        / selection_effects
                    )
                    current_gw_idx += 1
            assert current_gw_idx == n_gw
        # GWs share the same set of host galaxies
        else:
            selection_effects = self.selection_effects(H0_array, z, z_gal)
            for i, gw_dl in enumerate(gw_dl_array):
                likelihood_matrix[i, :] = (
                    self.gw_likelihood_array(gw_dl, H0_array, z, z_gal)
                    / selection_effects
                )

        return combine_posteriors(likelihood_matrix, H0_array)
