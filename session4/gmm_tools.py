#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def generate_Bayes_adventures_data(N=500, D=2):
    
    pi = [1/3, 1/3, 1/3]
    #mu = [np.array([2, 5]), np.array([4, 5]), np.array([6, 5])]
    #Sigma = [np.array([[0.5, 0], [0, 1]]), np.array([[0.1, 0], [0, 0.1]]), np.array([[0.1, 0], [0, 0.7]])]

    mu = [np.array([2, 5]), np.array([4, 4]), np.array([5.5, 4])]
    Sigma = [np.array([[0.2, -0.25], [-0.25, 1]]), np.array([[0.5, 0.3], [0.3, 1]]), np.array([[0.3, -0.2], [-0.2, 0.5]])]

    z = np.zeros(N, dtype=int)
    x = np.zeros((N, D))

    for n in np.arange(N):
        z[n] = np.argmax(np.random.multinomial(1, pi))
        x[n,:] = np.random.multivariate_normal(mu[z[n]], Sigma[z[n]])
        
    return pi, mu, Sigma, z, x

def plot_gaussian(mean, covar, color='b', ax=None):
    # adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html

    v, w = sp.linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / sp.linalg.norm(w[0])

    # Plot an ellipse to show the Gaussian component
    if ax is None:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
    
    ax.scatter(mean[0], mean[1], marker='o', edgecolors='k', color=color, s=75)

    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees

    ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color, linewidth=2)
    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('none')

def plot_data(x, ax=None):

    if ax is None:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)

    ax.scatter(x[:,0], x[:,1], s=10, color='b')
        
def plot_data_and_gaussians(x, means, covars, colors=['b','g','r'], ax=None):

    if ax is None:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        
    ax.scatter(x[:,0], x[:,1], s=10, color='k')

    for i, (mean, covar, color) in enumerate(zip(
            means, covars, colors)):

        plot_gaussian(mean, covar, color, ax)

def plot_GMM(x, z, means, covars, colors=['b','g','r'], ax=None):
    
    if ax is None:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        
    for k in np.arange(len(means)):
        ax.scatter(x[z==k,0], x[z==k,1], s=10, color=colors[k])

    for i, (mean, covar, color) in enumerate(zip(
            means, covars, colors)):

        plot_gaussian(mean, covar, color, ax)