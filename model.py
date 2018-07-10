

""" 
Simultaneous latent factor analysis and clustering.
"""

import numpy as np
import warnings
from scipy.special import logsumexp
from scipy.spatial import distance
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import (fast_logdet, randomized_svd, row_norms,
                                   squared_norm)

SMALL = 1e-12


def _svd(X, **kwargs):
    _, s, V = randomized_svd(X, **kwargs)
    return (s, V, squared_norm(X) - squared_norm(s))


def _log_prob_gaussian(X, mu, sigma):

    log_det = np.sum(np.log(sigma**(-0.5)), axis=1)

    N, D = X.shape
    precision = 1.0/sigma
    log_prob = (np.sum((mu**2 * precision), 1) \
             - 2.0 * np.dot(X, (mu * precision).T) \
             + np.dot(X**2, precision.T))

    return -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det



def _factor_scores(X, W, psi):
    """
    Calculate the factor scores given the data, the factor loads, and the
    noise variances.

    :param X:
        The data (NxD).

    :param W:
        The factor loads.

    :param psi:
        The noise variances in each D dimension.

    :returns:
        The factor scores, :math:`\theta`.
    """

    J, D = W.shape
    W_psi = W / psi
    Z = np.linalg.inv(np.eye(J) + np.dot(W_psi, W.T))
    theta = np.dot(np.dot(X, W_psi.T), Z)

    return theta



def _kmeans_pp(X, K, regularization=1e-10, random_state=None):

    N, D = X.shape
    random_state = check_random_state(random_state)
    squared_norms = row_norms(X, squared=True)

    mu = cluster.k_means_._k_init(X, K,
                                  x_squared_norms=squared_norms,
                                  random_state=random_state)

    # Assign everything to the closest mixture.
    labels = np.argmin(distance.cdist(mu, X), axis=0)

    # Generate responsibility matrix.
    responsibility = np.zeros((K, N))
    responsibility[labels, np.arange(N)] = 1.0
    membership = np.sum(responsibility, axis=1)

    # Calculate weights.
    weight = membership / N

    # Estimate covariance matrices.
    sigma = np.zeros(K, dtype=float) # white noise in factor scores

    for k, (r, em) in enumerate(zip(responsibility, membership)):
        dn = em - 1 if em > 1 else em
        # TODO check all this
        sigma[k] = np.mean(np.dot(r, (X - mu)**2) / dn + regularization)

    return (mu, sigma, weight, responsibility)



class LatentFactorModel(object):

    def __init__(self, n_components=1, tol=1e-2, max_iter=1000,
                 noise_variance_init=None, iterated_power=3, random_state=0,
                 **kwargs):

        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.iterated_power = iterated_power
        self.random_state = random_state
        
        return None


    def fit(self, X):
        """
        Fit the latent factor model to the data.
        """

        X = np.copy(np.atleast_2d(X))
        assert X.shape[0] > X.shape[1]


        n_samples, n_features = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # some constant terms
        nsqrt = np.sqrt(n_samples - 1) # -1 for mml
        llconst = n_features * np.log(2. * np.pi) + n_components
        var = np.var(X, axis=0)

        if self.noise_variance_init is None:
            psi = np.ones(n_features, dtype=X.dtype)
        else:
            if len(self.noise_variance_init) != n_features:
                raise ValueError("noise_variance_init dimension does not "
                                 "with number of features : %d != %d" %
                                 (len(self.noise_variance_init), n_features))
            psi = np.array(self.noise_variance_init)

        loglike = []
        old_ll = -np.inf
        SMALL = 1e-12

        def svd(X):
            _, s, V = randomized_svd(X, n_components,
                                     n_iter=self.iterated_power)
            return (s, V, squared_norm(X) - squared_norm(s))

        for i in range(self.max_iter):
            # SMALL helps numerics
            sqrt_psi = np.sqrt(psi) + SMALL
            s, V, unexp_var = svd(X / (sqrt_psi * nsqrt))
            s **= 2
            # Use 'maximum' here to avoid sqrt problems.
            W = np.sqrt(np.maximum(s - 1., 0.))[:, np.newaxis] * V
            del V
            W *= sqrt_psi

            # loglikelihood
            ll = llconst + np.sum(np.log(s))
            ll += unexp_var + np.sum(np.log(psi))
            ll *= -n_samples / 2.
            loglike.append(ll)
            if (ll - old_ll) < self.tol:
                break

            old_ll = ll

            psi = np.maximum(var - np.sum(W ** 2, axis=0), SMALL)

            
        else:
            warnings.warn('FactorAnalysis did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.',
                          ConvergenceWarning)

        self._factor_loads = W
        self._psi = psi
        self._log_likelihood = loglike
        self._n_iter = i + 1

        # Calculate factor scores.
        W_psi = W / psi
        Z = np.linalg.inv(np.eye(n_components) + np.dot(W_psi, W.T))
        self._factor_scores = np.dot(np.dot(X, W_psi.T), Z)

        return self

    @property
    def factor_loads(self):
        return self._factor_loads

    @property
    def factor_scores(self):
        return self._factor_scores

    @property
    def psi(self):
        return self._psi



class LatentClusteringModel(object):

    def __init__(self, n_components=1, n_clusters=1, tol=1e-2, max_iter=1000,
                 noise_variance_init=None, iterated_power=3, random_state=0,
                 **kwargs):

        self.n_components = n_components
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.iterated_power = iterated_power
        self.random_state = random_state
        
        return None


    def _check_array(self, X):
        X = np.copy(np.atleast_2d(X))
        if X.shape[0] <= X.shape[1]:
            raise ValueError("I don't believe you have more dimensions than data")
        return X



    def fit(self, X):

        X = self._check_array(X)

        N, D = X.shape
        J, K = (self.n_components, self.n_clusters)

        if J is None: 
            J = D

        if self.noise_variance_init is None:
            psi = np.ones(D, dtype=X.dtype)

        else:
            if len(self.noise_variance_init) != D:
                raise ValueError("noise_variance_init dimension does not "
                                 "with number of dimensions : %d != %d" %
                                 (len(self.noise_variance_init), D))
            psi = np.array(self.noise_variance_init)

        var = np.var(X, axis=0)
        self._mean = np.mean(X, axis=0)

        X -= self._mean

        nsqrt = np.sqrt(N - 1) # -1 for mml
        constant = -0.5 * N * (D * np.log(2. * np.pi) + J)

        lls = []
        previous_ll = -np.inf

        mu = np.zeros((K, J), dtype=float)
        cov = np.ones((K, J), dtype=float)
        weight = np.ones(K, dtype=float) / K

        W = np.zeros((J, D), dtype=float)
        R = np.zeros((N, K), dtype=float)

        for i in range(self.max_iter):

            # Maximization.
            W, S, uv = self._aecm_step_1(X, psi, nsqrt, J, self.iterated_power,
                                         mu, cov, R, W)

            # Update the noise variances.
            psi = np.maximum(var - np.sum(W**2, axis=0), SMALL)
     
            # Update our estimate of the factor scores.
            theta = _factor_scores(X, W, psi)

            # Calculate log-likelihood and responsibility matrix conditioned on
            # those parameter estimates.
            R, ll = self._expectation(X, W, psi, theta, mu, cov, weight, constant)

            # Update our estimates of the cluster means and cov, conditioned
            # on the estimate of theta and the responsibility matrix.
            mu, cov, weight = self._aecm_step_2(theta, R)

            # Calculate the log-likelihood 
            R, ll = self._expectation(X, W, psi, theta, mu, cov, weight, constant)

            print(i, ll)

            lls.append(ll)
            if np.abs(ll - previous_ll) < self.tol:
                break

            previous_ll = ll

        else:
            warnings.warn("Convergence not achieved: increase the maximum "\
                          "number of iterations", ConvergenceWarning)

        self._factor_loads = W
        self._factor_scores = theta
        self._noise_variance = psi
        self._cluster_mu = mu
        self._cluster_cov = cov
        self._cluster_weight = weight
        self._responsibility = R

        return self




    def _expectation(self, X, W, psi, theta, mu, cov, weight, constant=0):

        weighted_log_prob = np.log(weight) + _log_prob_gaussian(theta, mu, cov)

        log_likelihood = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

        responsibility = np.exp(log_responsibility)

        N, D = X.shape
        J, D = W.shape

        # Now calculate the log likelihood of the data.
        Q = (X + np.dot(np.dot(responsibility, mu), W)) \
          / (np.sqrt(N - 1) * (np.sqrt(psi) + SMALL))

        S, V, uv = _svd(Q, n_components=J, n_iter=self.iterated_power)

        log_likelihood = -0.5 * N \
                       * (np.sum(2 * np.log(S)) + uv + np.sum(np.log(psi))) \
                       + constant

        return (responsibility, log_likelihood)




    def _aecm_step_1(self, X, psi, nsqrt, n_components, iterated_power,
        mu, cov, R, W):
        """
        Perform step one of the alternating expectation maximization cycle,
        where we update our estimate of the latent factors.

        :param X:
            An array of data values.

        :param psi:
            The D-dimensional vector of noise variances.

        :param nsqrt:
            The correction factor for the number of samples. In maximum 
            likelihood this is :math:`\sqrt{N}` and in Minimum Message Length
            this is :math:`\sqrt{N - 1}`.

        :param n_components:
            The number of latent factor components.

        :param iterated_power:
            The iterated power to provide to SVD.
        """

        Q = (X + np.dot(np.dot(R, mu), W)) \
          / (nsqrt * (np.sqrt(psi) + SMALL))

        sqrt_psi = np.sqrt(psi) + SMALL
        S, V, unexplained_variance = _svd(Q,
                                          n_components=n_components,
                                          n_iter=iterated_power)
        S **= 2
        W = sqrt_psi * np.sqrt(np.maximum(S - 1., 0.))[:, np.newaxis] * V
        return (W, S, unexplained_variance)


    def _aecm_step_2(self, theta, responsibility, regularization=1e-10):
        """
        Perform step two of the alternating expectation conditional maximization
        cycle where we update our estimates of the mixture parameters in the
        factor scores, :math:`\theta`.
        """

        N, J = theta.shape

        N, K = responsibility.shape
        effective_membership = np.sum(responsibility, axis=0)

        # TODO: Check this against Kasarapu and Allison
        updated_weight = (effective_membership + 0.5)/(N + K/2.0)

        updated_mu = np.zeros((K, J), dtype=float)
        updated_cov = np.zeros((K, J), dtype=float) 

        for k, (r, em) in enumerate(zip(responsibility.T, effective_membership)):
            dn = em - 1 if em > 1 else em
            updated_mu[k] = np.sum(r * theta.T, axis=1) / dn
            updated_cov[k] = np.dot(r, (theta - updated_mu[k])**2) / dn \
                           + regularization

        return (updated_mu, updated_cov, updated_weight)



if __name__ == "__main__":

    from astropy.table import Table

    galah = Table.read("catalogs/GALAH_DR2.1_catalog.fits")


    label_names = [
        "fe_h",
        "na_fe",
        "mg_fe",
        "sc_fe",
        "ti_fe",
        "zn_fe",
        "mn_fe",
        "y_fe",
        "ca_fe",
        "ni_fe",
        "cr_fe",
        "o_fe",
        "si_fe",
        "k_fe",
        "ba_fe",
#        "eu_fe"
    ]

    y = np.array([galah[ln] for ln in label_names]).T

    passes_qc = np.ones(y.shape[0], dtype=bool)
    for label_name in label_names:
        if label_name == "fe_h": continue
        print(label_name, sum(passes_qc))
        passes_qc *= (galah["flag_{}".format(label_name)] == 0)

    y = y[passes_qc]
    N, D = y.shape

    # Subtract [Fe/H] to make everything [X/H]
    fe_h_index = label_names.index("fe_h")
    for d in range(D):
        if d == fe_h_index: continue
        y[:, d] += y[:, fe_h_index]

    label_names = ["{}_h".format(ea.split("_")[0]) for ea in label_names]

    print(N, D)
    model = LatentClusteringModel(n_components=3, n_clusters=5)
    model.fit(y)


