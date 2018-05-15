
// Factor Analysis model: The Empire Strikes Back

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    vector[D] y[N];
}

transformed data {
    int<lower=1> M; // number of non-zero loadings
    vector[D] mu;   // mean of the data in each dimension
    M = J * (D - J) + J * (J - 1)/2;
    mu = rep_vector(0.0, D);
}

parameters {
    vector<lower=0>[J] beta_diag;
    vector[M] beta_lower_triangular;
    vector<lower=0>[D] psi;

    real<lower=0> sigma_L;
}

transformed parameters {
    cov_matrix[D] Sigma;
    matrix[D, D] L_Sigma;
    matrix[D, J] L; // -> should be cholesky_factor_cov for speed

    {
        int idx = 0;
        for (i in 1:D) {
            for (j in (i + 1):J) {
                L[i, j] = 0;
            }
        }

        for (j in 1:J) {
            L[j, j] = beta_diag[j];
            for (i in (j + 1):D) {
                idx = idx + 1;
                L[i, j] = beta_lower_triangular[idx];
            }
        }

    }
    Sigma = multiply_lower_tri_self_transpose(L);
    for (i in 1:D)
        Sigma[i, i] = Sigma[i, i] + psi[i];
    L_Sigma = cholesky_decompose(Sigma);
}

model {
    // priors
    sigma_L ~ normal(0, 1);
    psi ~ normal(0, 1);
    beta_lower_triangular ~ normal(0, 1);

    // priors for diagonal entries to remain ~orthogonal and order invariant 
    //(Leung and Drton 2016)
    for (i in 1:J)
        target += (J - i) * log(beta_diag[i]) - 0.5 * beta_diag[i]^2 / sigma_L;

    y ~ multi_normal_cholesky(mu, L_Sigma);
}