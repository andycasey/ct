
import numpy as np


def generate_data(N, D, J, K=1, seed=None, full_output=False, **kwargs):
    """
    Generate data for :math:`J` dimensional latent factor model with :math:`D`
    dimensional observation vector of :math:`N` observations, clustered into
    :math:`K` clusters.
    """

    assert D == 10 and J == 3, \
        "Sorry these are fixed until we generate latent factors randomly"
    if seed is not None:
        np.random.seed(seed)

    mu_theta = np.zeros(J)
    mu_epsilon = np.zeros(D)

    phi = np.eye(J)
    psi_scale = kwargs.get("__psi_scale", 1)
    psi = np.diag(np.abs(np.random.normal(0, 1, size=D))) * psi_scale

    # TODO: generate latent factors randomly... but keep near orthogonality
    L = np.array([
        [0.99, 0.00, 0.25, 0.00, 0.80, 0.00, 0.50, 0.00,  0.00,  0.00],
        [0.00, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30],
        [0.00, 0.00, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00,  0.80,  0.80]
    ])

    if K == 1:
        theta = np.random.multivariate_normal(mu_theta, phi, size=N)
        responsibility = np.ones(N)

    else:
        # Calculate number of members per cluster.
        p = np.abs(np.random.normal(0, 1, K))
        responsibility = np.random.choice(np.arange(K), N, p=p/p.sum())

        cluster_mu_theta = np.random.multivariate_normal(
            np.zeros(J), np.eye(J), size=K)

        #S = 1 if kwargs.get("__cluster_common_scale", True) else J
        #cluster_mu_sigma = np.abs(np.random.multivariate_normal(
        #    np.zeros(J), 
        #    cluster_scale * np.random.normal(0, 1, size=S) * np.eye(J), 
        #    size=K))

        scale = kwargs.get("__cluster_scale", 1)
        cluster_sigma_theta = scale * np.abs(np.random.normal(0, 1, size=K))

        theta = np.zeros((N, J), dtype=float)
        for k, (mu, cov) in enumerate(zip(cluster_mu_theta, cluster_sigma_theta)):
            members = (responsibility == k)
            theta[members] = np.random.multivariate_normal(mu, np.eye(J) * cov, 
                                                           size=sum(members))
        

    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N)

    y = np.dot(theta, L) + epsilon
    data = dict(J=J, N=N, D=D, K=K, y=y)
    truths = dict(L=L, psi=psi, epsilon=epsilon, theta=theta, 
                  responsibility=responsibility)

    return (data, truths) if full_output else data





def simulate_data(N, D, J, K=1, seed=None, full_output=False, **kwargs):
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

    data = dict(J=J, N=N, D=D)

    if K == 1:
        theta = np.random.multivariate_normal(mu_theta, phi, size=N)

    else:
        # Calculate number of members per cluster.
        p = np.abs(np.random.normal(0, 1, K))
        responsibilities = np.random.choice(np.arange(K), N, p=p/p.sum())

        cluster_scale = kwargs.get("cluster_scale", 1)
        cluster_mu_theta = np.random.multivariate_normal(
            mu_theta, cluster_scale * np.abs(np.random.normal(0, 1, size=J)) * np.eye(J),
            size=K)
        cluster_mu_sigma = np.abs(
            np.random.multivariate_normal(np.zeros(J), np.eye(J), size=K))

        theta = np.zeros((N, J), dtype=float)
        for k, (mu, cov) in enumerate(zip(cluster_mu_theta, cluster_mu_sigma)):
            members = (responsibilities == k)
            theta[members] = np.random.multivariate_normal(mu, np.eye(J) * cov, 
                                                           size=sum(members))
            
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
