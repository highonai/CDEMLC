import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, multivariate_normal

class SemiparametricGaussianCopula:
    def __init__(self, k=100):
        self.k = k  # number of samples for KDE (nonparametric estimation)
    
    def fit_marginals(self, data):
        """Estimate the marginals using nonparametric KDE."""
        self.kde_marginals = []
        for i in range(data.shape[1]):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data[:, i].reshape(-1, 1))
            self.kde_marginals.append(kde)
    
    def transform_to_uniform(self, data):
        """Transform data to uniform [0,1] using estimated marginals."""
        u_data = np.zeros_like(data)
        for i, kde in enumerate(self.kde_marginals):
            log_density = kde.score_samples(data[:, i].reshape(-1, 1))
            cdf = np.exp(log_density.cumsum())  # Estimating the CDF from the KDE
            u_data[:, i] = cdf
        return u_data
    
    def fit_gaussian_copula(self, u_data):
        """Fit Gaussian copula to uniform marginals by estimating correlation matrix."""
        # Inverse CDF (quantile) transform to standard normal
        z_data = norm.ppf(u_data)
        # Estimate covariance matrix of the Z data (Gaussian copula correlation structure)
        self.corr_matrix = np.corrcoef(z_data, rowvar=False)
    
    def sample(self, n_samples=100):
        """Sample from the semiparametric Gaussian copula model."""
        mvn_samples = multivariate_normal(mean=np.zeros(self.corr_matrix.shape[0]), cov=self.corr_matrix).rvs(size=n_samples)
        u_samples = norm.cdf(mvn_samples)  # Transform back to uniform
        # Inverse KDE for each marginal to get original scale samples
        samples = np.zeros_like(u_samples)
        for i, kde in enumerate(self.kde_marginals):
            samples[:, i] = self.inverse_kde(u_samples[:, i], kde)
        return samples
    
    def inverse_kde(self, u, kde):
        """Inverse of the estimated KDE marginal to map uniform samples to original space."""
        x_vals = np.linspace(-5, 5, self.k)  # Sample points
        log_density = kde.score_samples(x_vals.reshape(-1, 1))
        cdf = np.exp(log_density.cumsum())  # Approximated CDF from KDE
        return np.interp(u, cdf, x_vals)  # Map uniform back to original
    
# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    data = np.random.rand(100, 3)  # 3-dimensional data
    
    # Initialize and fit the semiparametric Gaussian copula model
    model = SemiparametricGaussianCopula()
    model.fit_marginals(data)
    
    # Transform data to uniform [0,1]
    u_data = model.transform_to_uniform(data)
    
    # Fit Gaussian copula to uniform data
    model.fit_gaussian_copula(u_data)
    
    # Sample from the model
    samples = model.sample(n_samples=10)
    print(samples)
