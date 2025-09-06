import torch
import math


def phi_inv(u):
   
    eps = 1e-6
    u = u.clamp(eps, 1 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

def phi(z):

    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class EmpiricalCDF:
   
    def __init__(self, x: torch.Tensor):
        
        assert x.dim() == 1
        n = x.shape[0]
       
        v, idx = torch.sort(x)
        
        probs = (torch.arange(1, n + 1, dtype=torch.float32, device=x.device) / (n + 1.0))
        self.v = v
        self.p = probs

    def cdf(self, xq: torch.Tensor):
      
        shape = xq.shape
        x_flat = xq.reshape(-1)
       
        idx = torch.searchsorted(self.v, x_flat)

        n = self.v.numel()
       
        idx_lo = (idx - 1).clamp(0, n - 1)
        idx_hi = idx.clamp(0, n - 1)
        v_lo = self.v[idx_lo]
        v_hi = self.v[idx_hi]
        p_lo = self.p[idx_lo]
        p_hi = self.p[idx_hi]
       
        denom = (v_hi - v_lo)
        denom_zero = denom == 0
        t = torch.zeros_like(x_flat)
        nonzero = ~denom_zero
        t[nonzero] = (x_flat[nonzero] - v_lo[nonzero]) / denom[nonzero]
        
        u_flat = p_lo * (1 - t) + p_hi * t
        u = u_flat.reshape(shape)

        return u.clamp(1e-6, 1 - 1e-6)

    def inv(self, uq: torch.Tensor):
       
        shape = uq.shape
        u_flat = uq.reshape(-1)

        idx = torch.searchsorted(self.p, u_flat)
        n = self.p.numel()
        idx_lo = (idx - 1).clamp(0, n - 1)
        idx_hi = idx.clamp(0, n - 1)
        p_lo = self.p[idx_lo]
        p_hi = self.p[idx_hi]
        v_lo = self.v[idx_lo]
        v_hi = self.v[idx_hi]
        denom = (p_hi - p_lo)
        denom_zero = denom == 0
        t = torch.zeros_like(u_flat)
        nonzero = ~denom_zero
        t[nonzero] = (u_flat[nonzero] - p_lo[nonzero]) / denom[nonzero]
        x_flat = v_lo * (1 - t) + v_hi * t
        return x_flat.reshape(shape)


class SemiParametricGaussianCopula:
   
    def __init__(self, device='cpu', jitter=1e-6):
        self.device = torch.device(device)
        self.jitter = jitter

        self.n = None
        self.k = None  
        self.marginals = {}  
        self.corr = None  
        self.Sigma = None 
        self.fitted = False

    def fit(self, Y: torch.Tensor, F: torch.Tensor):
        """
        Fit empirical marginals and Gaussian copula correlation matrix.

        Y: (n,) tensor
        F: (n, K) tensor
        """
        assert Y.dim() == 1
        assert F.dim() == 2
        n, K = F.shape
        assert Y.shape[0] == n
        self.n = n
        self.k = K
        device = self.device

       
        self.marginals = {}
        self.marginals['y'] = EmpiricalCDF(Y.to(device))
        for j in range(K):
            self.marginals[f'f{j}'] = EmpiricalCDF(F[:, j].to(device))

       
        data_z = torch.zeros((n, K + 1), dtype=torch.float32, device=device)

        u_y = self.marginals['y'].cdf(Y.to(device))
        data_z[:, 0] = phi_inv(u_y)

        for j in range(K):
            uj = self.marginals[f'f{j}'].cdf(F[:, j].to(device))
            data_z[:, j + 1] = phi_inv(uj)



        z_mean = data_z.mean(dim=0, keepdim=True)
        z_centered = data_z - z_mean

        cov = (z_centered.t() @ z_centered) / (n - 1)

        std = cov.diag().clamp(min=1e-10).sqrt()
        corr = cov / (std[:, None] * std[None, :])

        corr = 0.5 * (corr + corr.t())

        self.corr = corr
        self.Sigma = cov  
        self.z_mean = z_mean.reshape(-1)  
        self.fitted = True

    def predict_conditional(self, f_query: torch.Tensor, method='mc', mc_samples=200, return_std=False):
       
        assert self.fitted, "Call fit() first."
        device = self.device
        K = self.k


        if f_query.dim() == 1:
            f_query = f_query.unsqueeze(0)
        m = f_query.shape[0]
        assert f_query.shape[1] == K


        z_f = torch.zeros((m, K), dtype=torch.float32, device=device)
        for j in range(K):
            u_j = self.marginals[f'f{j}'].cdf(f_query[:, j].to(device))
            z_f[:, j] = phi_inv(u_j)



        Sigma = self.Sigma
        Sigma_yy = Sigma[0, 0].unsqueeze(0) 
        Sigma_yf = Sigma[0, 1:].unsqueeze(0)  
        Sigma_fy = Sigma[1:, 0].unsqueeze(1)  
        Sigma_ff = Sigma[1:, 1:]  

       
        jitter = self.jitter * torch.eye(K, device=device)
        Sigma_ff_inv = torch.linalg.inv(Sigma_ff + jitter)

        
        A = Sigma_yf @ Sigma_ff_inv 
        mu_cond = (A @ z_f.t()).squeeze(0)  
        var_cond = (Sigma_yy - (A @ Sigma_fy).squeeze()).clamp(min=1e-12) 

        if method == 'plugin':
            
            u_pred = phi(mu_cond)
            y_hat = self.marginals['y'].inv(u_pred)
            if return_std:
                return y_hat, torch.zeros_like(y_hat)
            return y_hat

        elif method == 'mc':
           
            eps = torch.randn((mc_samples, m), device=device)
            z_samples = mu_cond.unsqueeze(0) + torch.sqrt(var_cond) * eps
            u_samples = phi(z_samples)  # (mc_samples, m)
          
            y_samples = []
            for i in range(m):
                y_i = self.marginals['y'].inv(u_samples[:, i])
                y_samples.append(y_i)
           
            y_samples = torch.stack(y_samples, dim=1)
            y_mean = y_samples.mean(dim=0)
            y_std = y_samples.std(dim=0, unbiased=False)
            if return_std:
                return y_mean, y_std
            return y_mean
        else:
            raise ValueError("method must be 'plugin' or 'mc'")

