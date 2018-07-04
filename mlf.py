
import numpy as np
import matplotlib.pyplot as plt

import stan_utils as stan
from data_utils import (simulate_data, test_data)
from mpl_utils import mpl_style

plt.style.use(mpl_style)


model = stan.load_stan_model("mlf.stan")
data, theta, epsilon, psi, L = test_data(True)
 
op_kwds = dict(init_alpha=1, tol_obj=1e-16, tol_rel_grad=1e-16, 
    tol_rel_obj=1e-16)
op_kwds = dict(data=data, seed=419906896)

s_opt = model.optimizing(**op_kwds)


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



if len(psi.shape) > 1:
    psi = np.diag(psi)

y = data["y"]
N, D = y.shape

factor_loads = s_opt["L"].T

b = factor_loads/np.sqrt(s_opt["psi"])
scaled_y = y/np.sqrt(s_opt["psi"])

b_sq = np.sum(b**2, axis=1)
N, J = theta.shape
factor_scores = np.dot(scaled_y, b.T) * (1 - J/(N - 1) * b_sq)/(1 + b_sq)

fig, axes = plt.subplots(J, figsize=(4, 12))
for j, ax in enumerate(axes):
    ax.scatter(theta.T[j], factor_scores.T[j])

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([np.min(limits), np.max(limits)])
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_xlabel(r"$J_{{{0},\textrm{{true}}}}$".format(j))
    ax.set_ylabel(r"$J_{{{0},\textrm{{inferred}}}}$".format(j))

axes[0].set_title(r"\textrm{factor scores}")
fig.tight_layout()
    


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


print("Done")

K = 2
fig, axes = plt.subplots(K, K, figsize=(10, 10))

axes[1, 0].scatter(factor_scores.T[0], factor_scores.T[1])
axes[1, 0].set_xlabel(r"$J_{{{0}}}$".format(0))
axes[1, 0].set_ylabel(r"$J_{{{0}}}$".format(1))

axes[0, 0].scatter(factor_scores.T[0], factor_scores.T[2])
axes[0, 0].set_xlabel(r"$J_{{{0}}}$".format(0))
axes[0, 0].set_ylabel(r"$J_{{{0}}}$".format(2))

axes[0, 1].set_visible(False)
axes[1, 1].scatter(factor_scores.T[2], factor_scores.T[1])
axes[1, 1].set_xlabel(r"$J_{{{0}}}$".format(2))
axes[1, 1].set_ylabel(r"$J_{{{0}}}$".format(1))

fig.tight_layout()

assert 0

# Let's try with N ~ 1e6 --> works in a minute or two.

from time import time
model = stan.load_stan_model("mlf.stan")
data, theta, epsilon, psi, L = simulate_data(N=1000000, J=3, D=10, seed=123,
                                             full_output=True)

op_kwds = dict(verbose=True, init_alpha=1, tol_obj=1e-16, tol_rel_grad=1e-16, 
    tol_rel_obj=1e-16)
op_kwds = dict(data=data, seed=419906896)

t_init = time()
s_opt = model.optimizing(**op_kwds)
t_elapsed = time() - t_init

print("Time elapsed: {:.0f}s".format(t_elapsed))


fig, ax = plt.subplots()
#ax.scatter(np.diag(psi), s_opt["psi"], facecolor="b")
ax.scatter(np.diag(psi), s_opt["psi"], facecolor="r")
ax.set_title("psi (full)")
limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
limits = np.array([limits.min(), limits.max()])
ax.plot(limits, limits, c="#666666", zorder=-1)
ax.set_xlim(limits)
ax.set_ylim(limits)


fig, axes = plt.subplots(L.shape[0])
for i, ax in enumerate(axes):
    ax.scatter(L[i], s_opt["L"].T[i], facecolor="r")

    ax.set_title("L (full)")

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([limits.min(), limits.max()])
    ax.plot(limits, limits, c="#666666", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)



if len(psi.shape) > 1:
    psi = np.diag(psi)

y = data["y"]
N, D = y.shape

factor_loads = s_opt["L"].T

b = factor_loads/np.sqrt(s_opt["psi"])
scaled_y = y/np.sqrt(s_opt["psi"])

b_sq = np.sum(b**2, axis=1)
N, J = theta.shape
factor_scores = np.dot(scaled_y, b.T) * (1 - J/(N - 1) * b_sq)/(1 + b_sq)

fig, axes = plt.subplots(J)
for j, ax in enumerate(axes):
    ax.scatter(theta.T[j], factor_scores.T[j], s=1, alpha=0.9, rasterized=True)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([np.min(limits), np.max(limits)])
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)


assert 0 # plotting gets expensive from here. should switch to logarithmic 2D histogram


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



