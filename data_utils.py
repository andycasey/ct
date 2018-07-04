
import numpy as np


def simulate_data(N, D, J, K=1, seed=None, full_output=False):
    """
    Simultae data for :math:`J` dimensional latent factor model with :math:`D`
    dimensional observation vector of :math:`N` observations.
    """

    if seed is not None:
        np.random.seed(seed)

    mu_theta = np.zeros(J)
    mu_epsilon = np.zeros(D)

    phi = np.eye(J)
    psi = np.diag(np.abs(np.random.normal(0, 1, size=D)))

    # TODO: generate latent factors randomly... but keep near orthogonality
    L = np.array([
        [0.99, 0.00, 0.25, 0.00, 0.80, 0.00, 0.50, 0.00,  0.00,  0.00],
        [0.00, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30],
        [0.00, 0.00, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00,  0.80,  0.80]
    ])

    if K == 1:
        theta = np.random.multivariate_normal(mu_theta, phi, size=N)

    else:
        # Calculate number of members per cluster.
        # TODO: assuming ~uniform weights.
        M = np.random.multinomial(N, np.ones(K, dtype=float)/K)

        theta = np.zeros((N, D), dtype=float)
        for i, m in enumerate(M):
            theta_cluster = np.random.normal(mu_theta, phi)

            raise NotImplementedError

            
    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N)

    y = np.dot(theta, L) + epsilon

    data = dict(y=y, J=J, N=N, D=D)

    if full_output:
        return (data, theta, epsilon, psi, L)
    else:
        return data






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
    #L = np.array([
    #    [0.99, 0.00, 0.25, 0.00, 0.80, 0.00, 0.50, 0.00, 0.00, 0.00],
    #    [-0.10, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30],
    #    [+0.30, 0.20, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00, 0.80, 0.80]
    #])

    np.random.seed(seed)

    theta = np.random.multivariate_normal(mu_theta, phi, size=N)
    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N)

    y = np.dot(theta, L) + epsilon

    data = dict(y=y, J=J, N=N, D=D)

    if full_output:
        return (data, theta, epsilon, psi, L)
    return data


def em_test_data(full_output=False):
    """
    Test data for checking expectation-maximization after the standard
    optimisation is run.
    """
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
        [-1, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30],
        [+1, -1, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00, 0.80, 0.80]
    ])

    np.random.seed(seed)

    theta = np.random.multivariate_normal(mu_theta, phi, size=N)
    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N)

    y = np.dot(theta, L) + epsilon

    data = dict(y=y, J=J, N=N, D=D)

    if full_output:
        return (data, theta, epsilon, psi, L)
    return data
