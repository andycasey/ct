

""" 
Simultaneous latent factor analysis and clustering.
"""

import numpy as np
import warnings
from sklearn.utils.extmath import fast_logdet, randomized_svd, squared_norm


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


