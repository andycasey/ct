
// Latent factor model that assumes homoscedastic noise on the data

data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimensionality of the data 
    int<lower=1> J; // number of latent factors
    vector[D] y[N]; // the data
}

transformed data {
    vector[D] mu;   // mean of the data in each dimension
    int<lower=1> M; // number of non-zero loadings

    M = J * (D - J) + choose(J, 2);
    mu = rep_vector(0.0, D);
}

parameters {
    //vector[M] beta_lower_triangular;
    //vector<lower=0>[J] beta_diag;
    vector<lower=0>[D] psi;
    real<lower=0> sigma_L;
    cov_matrix[D] Sigma;
    cholesky_factor_cov[D] L_Sigma;
}
transformed parameters{
    matrix[D, D] Psi;
    vector[J] beta_diag;
    Psi = rep_matrix(0.0, D, D);
    for (i in 1:D)
        Psi[i, i] = psi[i];



    for (j in 1:J) {
        beta_diag[j] = cholesky_decompose(Sigma)[j, j];
    }

}

model {
    // priors
    //beta_lower_triangular ~ normal(0, sigma_L);
    sigma_L ~ normal(0, 1);
    psi ~ normal(0, 1);

    // Priors for diagonal entries to remain ~orthogonal and order invariant 
    // (Leung and Drton 2016)
    for (i in 1:J)
       target += (J - i) * log(beta_diag[i]) - 0.5 * beta_diag[i]^2 / sigma_L;

    y ~ multi_normal(mu, Sigma + Psi);
}