
"""
Utilities for multiple latent factor models.
"""

import numpy as np
from sklearn.decomposition import FactorAnalysis


def factor_scores(y, factor_loads, psi):
    """
    Return the factor scores given the data, :math:`y`, the factor loads, and
    the noise variance in each dimension, :math:`\psi`.

    :param y:
        The data, which is expected to have shape (N, D), where N is the number
        of samples and D is the number of dimensions.

    :param factor_loads:
        An array of the factor loads, which is expected to have shape (D, J) 
        where D is the number of dimensions and J is the number of latent
        factors.

    :param psi:
        An array of noise variance in each D-dimension.

    :returns:
        A matrix of factor scores :math:`\theta` with shape (N, J).
    """

    psi = np.atleast_1d(psi)
    factor_loads = np.atleast_2d(factor_loads)
    D, J = factor_loads.shape

    W = factor_loads.T / psi
    Z = np.linalg.inv(np.eye(J) + np.dot(W, factor_loads))
    theta = np.dot(np.dot(y, W.T), Z)

    return theta


def initialization_point(y, J):
    """
    Run factor analysis to get a reasonable initialization point for the
    optimisation process.

    :param y:
        An array of the data that has shape (N, D) where N is the number of
        stars and D is the dimensionality of the data.

    :param J:
        The number of latent factors.

    :returns:
        A dictionary of initial values that can be fed directly to Stan.
    """

    fa = FactorAnalysis(J)
    fa.fit(y)

    # TODO: Re-order the matrix of elements such that the low absolute values 
    #       are in the upper triangular part of the matrix, and that the entries
    #       along the diagonal are positive.

    N, D = y.shape

    L, psi = (fa.components_.T, fa.noise_variance_)

    # The beta diagonal values must be positive.
    beta_diag = np.clip(L.T[np.diag_indices(J)], 0, np.inf) + 1e-3

    # A hack to get the lower triangular beta values is to set the upper
    # triangular (including the diagonal) to non-finite values then re-order
    # and flatten the array.
    beta_lower_triangular = np.copy(L)
    beta_lower_triangular[np.triu_indices_from(L, 0)] = np.nan
    beta_lower_triangular = beta_lower_triangular.T.flatten()
    _ = np.isfinite(beta_lower_triangular)
    beta_lower_triangular = beta_lower_triangular[_]

    sigma_L = np.std(beta_lower_triangular)

    init = dict(psi=psi,
                beta_diag=beta_diag,
                beta_lower_triangular=beta_lower_triangular,
                sigma_L=sigma_L)

    return init



def rank_order_triu_matrix(C, k=1):
    """
    Return the row and column indices for the matrix C that will result in the
    lowest absolute entries in C being in the upper triangular matrix. This
    utility function is for re-ordering latent factor vectors in order to
    approximate the factor loads as a lower triangular matrix, which enables us
    to use Cholesky decomposition tricks during optimization.

    :param C:
        A full-rank matrix of shape D x J where D is the number of dimensions
        in the data and J is the number of latent factors.

    :param k: [optional]
        The diagonal offset to apply when determining the number of near-zero
        entries for the upper triangular component of the matrix. By default
        :math:`k = 1`.

    :returns:
        A two-part tuple containing the row indices and column indices to
        re-order the matrix.
    """

    Q = max(np.triu_indices_from(C, k=1)[1])

    ci, cj = (np.argsort(C), np.argsort(C.T))

    col_cumsum = np.array([np.cumsum(c.T[i]) for c, i in zip(C.T, cj)])
    col_indices = np.argsort(col_cumsum.T[Q])[::-1]
    
    C = np.copy(C).T[col_indices].T
    row_cumsum = np.array([np.cumsum(c[i]) for c, i in zip(C, ci)])

    row_indices = np.argsort(row_cumsum.T[Q])

    return (row_indices, col_indices)


def atomic_number(label_names):

    periodic_table = """H                                                  He
                    Li Be                               B  C  N  O  F  Ne
                    Na Mg                               Al Si P  S  Cl Ar
                    K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                    Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                    Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                    Fr Ra Lr Rf"""

    lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
    actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"

    periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
                                   .replace(" Ra ", " Ra " + actinoids + " ") \
                                   .split()

    Z = 1 + np.array([periodic_table.index(ln.split("_")[0].title()) \
                        for ln in label_names])
    return Z

    