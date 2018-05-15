
// Bayesian Factor Analysis model

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    matrix[N, D] y;
}

transformed data {
    int<lower=1> M; // number of non-zero loadings
    vector[D] mu;   // mean of the data in each dimension
    M = J * (D - J) + J * (J - 1)/2;
    mu = rep_vector(0.0, D);
}

parameters {
    vector[M] L_t;  // lower diagonal elements of L
    vector<lower=0>[J] L_d; // lower diagonal elements of L
    vector<lower=0>[D] psi; // vector of variances
    real<lower=0> mu_psi;
    real<lower=0> sigma_psi;
    real mu_lt;
    real<lower=0> sigma_lt;
}

transformed parameters {
    cholesky_factor_cov[D, J] L; // lower triangular factor loadings
    cov_matrix[D] Q; // covariance matrix
    {
        int idx1;
        int idx2;
        idx1 = 0;
        idx2 = 0;
        for (i in 1:D) {
            for (j in (i + 1):J) {
                idx1 = idx1 + 1;
                L[i, j] = 0; // constrain upper triangular elements to zero
            }
        }

        for (j in 1:J) {
            L[j, j] = L_d[j];
            for (i in (j + 1):D) {
                idx2 = idx2 + 1;
                L[i, j] = L_t[idx2];
            }
        }
    }

    Q = L * L' + diag_matrix(psi);
}

model {
    // hyperpriors
    mu_psi ~ cauchy(0, 1);
    sigma_psi ~ cauchy(0, 1);
    mu_lt ~ cauchy(0, 1);
    sigma_lt ~ cauchy(0, 1);

    // priors
    L_d ~ cauchy(0, 3);
    L_t ~ cauchy(mu_lt, sigma_lt);
    psi ~ cauchy(mu_psi, sigma_psi);

    for (j in 1:N)
        y[j] ~ multi_normal(mu, Q);
}