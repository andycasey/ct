
import numpy as np

import stan_utils as stan


def simulate_data(N, D, J, seed=None):
    """
    Simultae data for :math:`J` dimensional latent factor model with :math:`D`
    dimensional observation vector of :math:`N` observations.
    """

    if seed is not None:
        np.random.seed(seed)

    mu_theta = np.zeros(J)
    mu_epsilon = np.zeros(D)

    phi = np.eye(J)

    raise NotImplementedError("check distributions")



def test_data(full_output=False):

    seed = 123

    J = 3
    D = 10
    N = 300

    mu_theta = np.zeros(J)
    mu_epsilon = np.zeros(D)

    phi = np.eye(J)
    psi = np.diag([
        0.2079, 0.19, 0.1525, 0.20, 0.36, 0.1875, 0.1875, 1.00, 0.27, 0.27])

    L = np.array([
        [0.99, 0.00, 0.25, 0.00, 0.80, 0.00, 0.50, 0.00, 0.00, 0.00],
        [0.00, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30],
        [0.00, 0.00, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00, 0.80, 0.80]
    ])

    np.random.seed(seed)

    theta = np.random.multivariate_normal(mu_theta, phi, size=N)
    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N)

    y = np.dot(theta, L) + epsilon

    data = dict(y=y, J=J, N=N, D=D)

    if full_output:
        return (data, theta, epsilon, psi, L)
    return data


model = stan.load_stan_model("bfa.stan")

data = test_data()
s_opt = model.optimizing(data=data)

s_samples = model.sampling(**stan.sampling_kwds(data=data, init=s_opt, chains=2))

assert 0

# Converges:
sigma = 0.5
n_samples, n_features, n_clusters, rank = 1000, 50, 1, 1
rng = np.random.RandomState(123)


true_homo_specific_variances = sigma**2 * np.ones((1, n_features))


U, _, _ = np.linalg.svd(rng.randn(n_features, n_features))
true_factor_loads = U[:, :rank].T


true_factor_scores = rng.randn(n_samples, rank)
X = np.dot(true_factor_scores, true_factor_loads)

# Assign objects to different clusters.
indices = rng.randint(0, n_clusters, size=n_samples)
true_weights = np.zeros(n_clusters)
true_means = rng.randn(n_clusters, n_features)
for index in range(n_clusters):
    X[indices==index] += true_means[index]
    true_weights[index] = (indices==index).sum()

true_weights = true_weights/n_samples

# Adding homoscedastic noise
bar = rng.randn(n_samples, n_features)
X_homo = X + sigma * bar

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas
true_hetero_specific_variances = sigmas**2

data = X_hetero

model = slf.SLFGMM(n_clusters)
model.fit(data)
