
"""
Can we do stable optimisation using Stan/TensorFlow, and then run expectation-
maximisation from the optimised point in order to solve for the missing latent
factor load values?
"""

import numpy as np
import stan_utils as stan

from data_utils import em_test_data



model = stan.load_stan_model("mlf.stan")
data, theta, epsilon, psi, L = em_test_data(True)

 
op_kwds = dict(init_alpha=1, tol_obj=1e-16, tol_rel_grad=1e-16, 
    tol_rel_obj=1e-16)
op_kwds = dict(data=data, seed=419906896)

# Run the first optimisation step using Stan.
s_opt = model.optimizing(**op_kwds)

y = data["y"]
N, D = y.shape
N, J = theta.shape

factor_loads = s_opt["L"].T

b = factor_loads/np.sqrt(s_opt["psi"])
scaled_y = y/np.sqrt(s_opt["psi"])

b_sq = np.sum(b**2, axis=1)
factor_scores = np.dot(scaled_y, b.T) * (1 - J/(N - 1) * b_sq)/(1 + b_sq)

alt_factor_loads = np.copy(s_opt["L"].T)
alt_factor_loads[1, 0] = L[1, 0]
alt_factor_loads[2, 0] = L[2, 0]
alt_factor_loads[2, 1] = L[2, 1]

alt_b = alt_factor_loads/np.sqrt(s_opt["psi"])
alt_b_sq = np.sum(alt_b**2, axis=1)
alt_factor_scores = np.dot(scaled_y, alt_b.T) * (1 - J/(N - 1) * alt_b_sq)/(1 + alt_b_sq)


# Predict data.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5)
axes = np.array(axes).flatten()

faux_y = np.dot(factor_scores, factor_loads)
true_y = np.dot(theta, L)
alt_y = np.dot(alt_factor_scores,  alt_factor_loads)

for i, ax in enumerate(axes):
    ax.scatter(y.T[i], faux_y.T[i], s=1, alpha=0.5)
    ax.scatter(y.T[i], true_y.T[i], s=1, facecolor="#666666", zorder=-1, alpha=0.5)
    ax.scatter(y.T[i], alt_y.T[i], facecolor="r", alpha=0.5, s=1)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([limits.min(), limits.max()])
    ax.set_xlim(limits)
    ax.set_ylim(limits)

raise a


factor_loads = L
if len(psi.shape) > 1:
    psi = np.diag(psi)


# Let's try Wallace and Freeman 1990 method first.
y = data["y"]
N, D = y.shape
J = data["J"]

y = data["y"]
_psi = np.copy(s_opt["psi"])

var = np.var(y, axis=0)

from sklearn.utils.extmath import squared_norm

old_ll = None
for i in range(10000):

    _sqrt_psi = np.sqrt(_psi)

    _, s, V = np.linalg.svd(y / (_sqrt_psi * np.sqrt(N)))

    unexp_var = squared_norm(s[J:])
    s, V = s[:J], V[:J]

    s2 = s**2
    W = np.sqrt(np.maximum(s2 - 1, 0))[:, np.newaxis] * V
    W *= _sqrt_psi

    ll = np.sum(np.log(s2)) + np.sum(np.log(_psi)) + unexp_var

    print(i, ll)

    _psi = np.maximum(var - np.sum(W**2, axis=0), 1e-16)

    if old_ll is not None and (ll - old_ll) < 1e-10:
        break

    old_ll = ll

new_factor_loads = V #W * np.sqrt(_psi)





"""

raise a


w = y #- np.mean(y, axis=0)
V = np.dot(w, w.T)

factor_loads = s_opt["L"].T

b = factor_loads/np.sqrt(s_opt["psi"])

for i in range(1000):

    specific_variances = np.sum(w**2, axis=0) \
                       / ((N - 1) * (1 + np.sum(b**2, axis=0)))
    specific_sigmas = np.atleast_2d(np.sqrt(specific_variances))

    #b_sq = np.sum(b**2, axis=0)
    b_sq = np.sum(b**2)

    Y = np.corrcoef(w.T) / np.dot(specific_sigmas.T, specific_sigmas)


    #factor_scores = np.dot(scaled_y, b.T) * (1 - J/(N - 1) * b_sq)/(1 + b_sq)

    #(1 - J/(N - 1) * b_sq)/(1 + b_sq)

    new_beta = np.dot(Y, b.T).T * (1 - J/(N - 1) * b_sq)/((N - 1) * (1 + b_sq))
    #             / ((N - 1) * (1 + b_sq))
    #new_beta = new_beta.T

    change = np.sum(np.abs(b - new_beta))

    print(i, change)
    b = new_beta

    assert np.isfinite(change)

    if change < np.inf:
        break

specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
                / ((np.sum(b**2, axis=0) + 1.0) * (N - 1)))

new_factor_loads = specific_sigmas * b # np.sqrt(s_opt["psi"]) * b

#specific_variances = specific_sigmas**2



w = y# - np.mean(y, axis=0)
V = np.dot(w.T, w)

factor_loads = np.copy(s_opt["L"])
specific_variances = np.copy(s_opt["psi"])**2
beta = (factor_loads.T/np.sqrt(specific_variances))

for i in range(100):

    beta = beta.reshape((D, -1))

    
    #spec_v = np.atleast_2d(np.diag(V)).T / ((N - 1.0) * (1.0 + beta**2))
    spec_v = np.atleast_2d(np.sum(w**2, axis=0)).T / ((N - 1.0) * (1.0 + beta**2))
    spec_s = np.atleast_2d(spec_v)

    Y = V / np.dot(spec_s, spec_s.T)

    b_sq = np.sum(beta**2, axis=0)
    beta_new = (np.dot(Y, beta) * (1.0 - J/(N - 1.0) * b_sq)) \
             / ((N - 1.0) * (1.0 + b_sq))
    beta_new = beta_new

    l1_norm = np.abs(beta - beta_new).sum()

    print(i, l1_norm)

    assert np.isfinite(l1_norm)

    beta = beta_new

    if l1_norm < 1e-5:
        break

new_factor_loads = (np.sqrt(spec_v) * beta).reshape((-1, D)).T
"""



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(psi, s_opt["psi"], facecolor="r")
ax.scatter(psi, _psi, facecolor="b")

fig, axes = plt.subplots(J)


for j, ax in enumerate(axes):

    ax.scatter(L[j], factor_loads[j], facecolor="r")
    ax.scatter(L[j], new_factor_loads[j], facecolor="b")

    #ax.scatter(factor_loads.T[j], new_factor_loads.T[j])

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.array([np.min(limits), np.max(limits)])
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)




raise a

# OK, now run E-M from this position.



