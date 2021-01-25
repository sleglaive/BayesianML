import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats


def f(X, noise_variance, f_w0 = -0.3, f_w1 =  0.5):
    '''Linear function plus noise'''
    return f_w0 + f_w1 * X + noise(X.shape, noise_variance)


def g(X, noise_variance):
    '''Sinusoidial function plus noise'''
    return 0.5 + np.sin(2 * np.pi * X) + noise(X.shape, noise_variance)


def noise(size, variance):
    return np.random.normal(scale=np.sqrt(variance), size=size)

def identity_basis_function(x):
    return x


def gaussian_basis_function(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def polynomial_basis_function(x, power):
    return x ** power


def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)


def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    t_mse = Phi_test @ m_N
    # Only compute variances (diagonal elements of covariance matrix)
    sigma_N2 = 1 / beta + np.sum(Phi_test @ S_N * Phi_test, axis=1)
    
    return t_mse, sigma_N2

def compute_ELBO(a, b, c, d, a_tilde, b_tilde, c_tilde, d_tilde, mu_N, Sigma_N, t, Phi):
    
    N, M = Phi.shape

    exp_beta = c_tilde/d_tilde
    exp_log_beta = sp.special.digamma(c_tilde) - np.log(d_tilde)
    exp_alpha = a_tilde/b_tilde
    exp_log_alpha = sp.special.digamma(a_tilde) - np.log(b_tilde)
    exp_w_epsilon = np.sum( (t - Phi @ mu_N)**2 ) + np.trace(Phi @ Sigma_N @ Phi.T)
    exp_w = mu_N**2 + np.diag(Sigma_N)[:, np.newaxis]
    
    likelihood_term = 0.5*(N*exp_log_beta - exp_beta*exp_w_epsilon)
    w_term = 0.5*np.sum(exp_log_alpha - exp_alpha*exp_w)
    alpha_term = np.sum( (a-1)*exp_log_alpha - b*exp_alpha )
    beta_term = (c-1)*exp_log_beta - d*exp_beta
    
    entropy_w = 0.5*np.log(np.linalg.det(Sigma_N))
    entropy_alpha = np.sum(a_tilde - np.log(b_tilde) + np.log(sp.special.gamma(a_tilde)) 
                           + (1-a_tilde)*sp.special.digamma(a_tilde))
    entropy_beta = (c_tilde - np.log(d_tilde) + np.log(sp.special.gamma(c_tilde)) 
                    + (1-c_tilde)*sp.special.digamma(c_tilde))
    
    return likelihood_term + w_term + alpha_term + beta_term + entropy_w + entropy_alpha + entropy_beta

def plot_data(x, t):
    plt.scatter(x, t, marker='o', c="k", s=20)


def plot_truth(x, y, label='Truth'):
    plt.plot(x, y, 'k--', label=label)


def plot_predictive(x, y, std, y_label='Prediction', std_label='Uncertainty', plot_xy_labels=True):
    y = y.ravel()
    std = std.ravel()

    plt.plot(x, y, label=y_label)
    plt.fill_between(x.ravel(), y + std, y - std, alpha = 0.5, label=std_label)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior_samples(x, ys, plot_xy_labels=True):
    plt.plot(x, ys[:, 0], 'r-', alpha=0.5, label='Post. samples')
    for i in range(1, ys.shape[1]):
        plt.plot(x, ys[:, i], 'r-', alpha=0.5)

    if plot_xy_labels:
        plt.xlabel('x')
        plt.ylabel('y')


def plot_posterior(mean, cov, w0, w1):
    resolution = 100

    grid_x = grid_y = np.linspace(-1, 1, resolution)
    grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)
    plt.imshow(densities, origin='lower', extent=(-1, 1, -1, 1))
    plt.scatter(w0, w1, marker='x', c="r", s=20, label='Truth')

    plt.xlabel('w0')
    plt.ylabel('w1')


def print_comparison(title, a, b, a_prefix='np', b_prefix='br'):
    print(title)
    print('-' * len(title))
    print(f'{a_prefix}:', a)
    print(f'{b_prefix}:', b)
    print()