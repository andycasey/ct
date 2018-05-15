
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


model = stan.load_stan_model("bfa2.stan")
data, theta, epsilon, psi, L = test_data(True)

s_opt = model.optimizing(data=data, seed=419906896)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
#ax.scatter(np.diag(psi), s_opt["psi"], facecolor="b")
ax.scatter(np.diag(psi), s_opt["psi"], facecolor="r")
ax.set_title("psi")
limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
limits = np.array([limits.min(), limits.max()])
ax.plot(limits, limits, c="#666666", zorder=-1)
ax.set_xlim(limits)
ax.set_ylim(limits)


fig, axes = plt.subplots(L.shape[0])
for i, ax in enumerate(axes):
    ax.scatter(L[i], s_opt["L"].T[i], facecolor="r")

    ax.set_title("L")

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([limits.min(), limits.max()])
    ax.plot(limits, limits, c="#666666", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)


"""
specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
            / ((b*b + 1.0) * (N - 1)))
factor_loads = specific_sigmas * b
scaled_y = (data - w_means)/specific_sigmas

b_sq = np.sum(b**2)
factor_scores = np.dot(scaled_y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
specific_variances = specific_sigmas**2
"""

if len(psi.shape) > 1:
    psi = np.diag(psi)

y = data["y"]
N, D = y.shape

"""
b = L/np.sqrt(psi)
specific_sigmas = np.sqrt(np.diag(np.dot(y.T, y)) \
                / ((np.diag(np.dot(b.T, b)) + 1.0) * (N - 1)))

#np.sqrt(psi) = specific_sigmas

scaled_y = y/specific_sigmas

b_sq = np.sum(b**2, axis=1)
N, J = theta.shape
factor_scores = np.dot(scaled_y, b.T) * (1 - J/(N - 1) * b_sq)/(1 + b_sq)
"""
factor_loads = s_opt["L"].T

b = factor_loads/np.sqrt(s_opt["psi"])
scaled_y = y/np.sqrt(s_opt["psi"])

b_sq = np.sum(b**2, axis=1)
N, J = theta.shape
factor_scores = np.dot(scaled_y, b.T) * (1 - J/(N - 1) * b_sq)/(1 + b_sq)

fig, axes = plt.subplots(J)
for j, ax in enumerate(axes):
    ax.scatter(theta.T[j], factor_scores.T[j])

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([np.min(limits), np.max(limits)])
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)



fig, axes = plt.subplots(2, 5)
axes = np.array(axes).flatten()

faux_y = np.dot(factor_scores, factor_loads)
true_y = np.dot(theta, L)

for i, ax in enumerate(axes):
    ax.scatter(y.T[i], faux_y.T[i], alpha=0.5)
    ax.scatter(y.T[i], true_y.T[i], facecolor="#666666", zorder=-1, alpha=0.5)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([limits.min(), limits.max()])
    ax.set_xlim(limits)
    ax.set_ylim(limits)


raise a


factor_loads = specific_sigmas * b
scaled_y = (data - w_means)/specific_sigmas

b_sq = np.sum(b**2)
factor_scores = np.dot(scaled_y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
specific_variances = specific_sigmas**2


b = ((s_opt["L"].T)/np.sqrt(s_opt["psi"]))
#b = (np.dot(Y, b) * (1 - K/(N - 1) * b_sq)) \
#                  / ((N - 1) * (1 + b_sq))

specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
            / (b*b + 1.0) * (N - 1))

scaled_y = y/specific_sigmas

#spec_sig = np.atleast_2d(np.sqrt(np.diag(np.dot(y.T, y)))).T / ((b*b + 1.0) * (N - 1))
#scaled_y = y/np.sqrt(np.sum(spec_sig**2, axis=1))


b_sq = np.sum(b**2, axis=0)
N, D = y.shape
factor_scores = np.dot(scaled_y, b) * (1 - D/(N - 1) * b_sq)/(1 + b_sq)

if len(psi.shape) > 1:
    psi = np.diag(psi)


t_sy = y/np.sqrt(psi)

t_b = (L/np.sqrt(psi)).T
t_bsq = np.sum(t_b**2, axis=0)
t_fs = np.dot(t_sy, t_b) * (1 - D/(N - 1) * t_bsq)/(1 + t_bsq)

#factor_scores *= np.array([18.5, 13.5, 6.9])


N, J = theta.shape
fig, axes = plt.subplots(J)
for j, ax in enumerate(axes):
    ax.scatter(theta.T[j], factor_scores.T[j])
    ax.scatter(theta.T[j], t_fs.T[j], facecolor="#666666")

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([np.min(limits), np.max(limits)])
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)


raise a

fig, axes = plt.subplots(2, 5)
axes = np.array(axes).flatten()

faux_y = np.dot(factor_scores, factor_loads.T)
true_y = np.dot(theta, L)

for i, ax in enumerate(axes):
    ax.scatter(y.T[i], faux_y.T[i], alpha=0.5)
    ax.scatter(y.T[i], true_y.T[i], facecolor="#666666", zorder=-1, alpha=0.5)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([limits.min(), limits.max()])
    ax.set_xlim(limits)
    ax.set_ylim(limits)

#s_samples = model.sampling(**stan.sampling_kwds(data=data, init=s_opt, chains=2))

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
