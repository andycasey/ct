
import numpy as np
import stan_utils as stan


n_samples, n_features, n_clusters, rank = 300, 10, 1, 3
sigma = 0.5
true_homo_specific_variances = sigma**2 * np.ones((1, n_features))

rng = np.random.RandomState(123)

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


model = stan.load_stan_model("bfa2.stan")


data = dict(y=X_homo, N=n_samples, D=n_features, J=rank)
s_opt = model.optimizing(data=data)

print(s_opt)


import matplotlib.pyplot as  plt

scatter_kwds = dict(s=5, facecolor="#000000", rasterized=True)

fig, ax = plt.subplots()
ax.scatter(
    np.sqrt(true_homo_specific_variances.flatten()),
    s_opt["sigma_y"],
    **scatter_kwds)
ax.set_title("specific scatter")


L = s_opt["beta_diag"] * s_opt["L"]
fig, ax = plt.subplots()
ax.scatter(
    true_factor_loads.flatten(),
    s_opt["L"].T.flatten(),
    **scatter_kwds)
ax.set_title("factor loads")

