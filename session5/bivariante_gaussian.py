#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:00:35 2020

@author: sleglaive
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

x, y = np.mgrid[-1.55:1.55:0.01, -1.55:1.55:0.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y


sigma_r2 = np.array([0.5, 0.9, 0.1, 0.9])
sigma_i2 = np.array([0.5, 0.1, 0.9, 0.5])
rho = np.array([0, 0, 0, 0.4])


plt.figure(figsize=(10, 10))


for n in np.arange(4):

    plt.subplot(2,2,n+1)

    mu = [0, 0]
    Sigma = [
            [sigma_r2[n], rho[n]], 
            [rho[n], sigma_i2[n]]
            ]
    
    rv = multivariate_normal(mu, Sigma)
    
    plt.contour(x, y, rv.pdf(pos))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks(np.array([-2,0,2]), np.array([-2,0,2]))
    plt.yticks(np.array([-2,0,2]), np.array([-2,0,2]))
    plt.xlabel('$x_1$', fontsize=18)
    plt.ylabel('$x_2$', fontsize=18)
    plt.title(r'$(\sigma_1^2, \sigma_2^2, \rho) = ({:.1f}, {:.1f}, {:.1f})$'.format(sigma_r2[n], sigma_i2[n], rho[n]), fontsize=18)
    plt.grid('on')


plt.tight_layout()