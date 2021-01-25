import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def to_img(x):
    # convert a tensor vectorized image to a numpy image of shape 28 x 28
    if torch.is_tensor(x):
        x = x.cpu().data.numpy()
    x = x.reshape([-1, 28, 28])
    return x  
    
def plot_reconstructions_VAE(model, test_loader, device='cpu'):
    """
    Plot 10 reconstructions from the test set. The top row is the original
    digits, the bottom is the decoder reconstruction.
    The middle row is the encoded vector.
    """
    # encode then decode
    data, _ = next(iter(test_loader))
    data = data.view([-1, 784]) # the size -1 is inferred from other dimensions, shape (batch size, 784)
    data.requires_grad = False
    data = data.to(device)
    true_imgs = data
    encoded_imgs_mean, encoded_imgs_log_var = model.encode(data)
    encoded_imgs_sampled = model.reparameterize(encoded_imgs_mean, encoded_imgs_log_var)
    decoded_imgs = model.decode(encoded_imgs_sampled)
    
    true_imgs = to_img(true_imgs)
    decoded_imgs = to_img(decoded_imgs)
    encoded_imgs_sampled = encoded_imgs_sampled.cpu().data.numpy()
    
    n = 10
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(true_imgs[i], interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs_sampled[np.newaxis,i,:].T, interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i], interpolation="nearest", 
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()
    
def display_digits(X, digit_size=28, n_i=20, n_j=20, figsize=(20, 20)):
    
    figure = np.zeros((digit_size * n_i, digit_size * n_j))
    
    for i in range(n_i):
        for j in range(n_j):            
            x = i * digit_size
            y = j * digit_size
            figure[x:x + digit_size, y:y + digit_size] = X[i,j,:,:]
    
    plt.figure(figsize=figsize)
    plt.imshow(figure, cmap='Greys_r')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()

def illust_walk_latent_space():
    
    from scipy.stats import norm

    n_grid = 12

    uniform_samples = np.linspace(0.01,0.99,n_grid) # linearly spaced points between 0 and 1
    normal_samples = norm.ppf(uniform_samples) # map through inverse CDF

    plt.figure(figsize=(7,10))
    plt.subplot(2,1,1)

    # Plot CDF
    x_norm = np.linspace(normal_samples[0], -normal_samples[0], 100)
    plt.plot(x_norm, norm.cdf(x_norm), 'k-', lw=2, alpha=1)

    # Plot linearly spaced points between 0 and 1
    line = plt.plot(np.ones_like(normal_samples)*normal_samples[0], uniform_samples, 'o', markersize=10)[0]
    line.set_clip_on(False)

    # Plot points mapped through the inverse CDF
    line = plt.plot(normal_samples, np.zeros_like(normal_samples), 'o', markersize=10)[0]
    line.set_clip_on(False)

    plt.xlim([normal_samples[0], -normal_samples[0]])
    plt.ylim([0, 1])
    plt.title('cumulative distribution function (CDF) of N(0,1)', fontsize=15)
    plt.ylabel('F(z)', fontsize=20)
    plt.xlabel('z', fontsize=20)
    plt.legend({'CDF', 
                'linearly spaced points between 0 and 1', 
                'points mapped through the inverse CDF'}, loc='upper right', fontsize=10)

    # Plot PDF
    plt.subplot(2,1,2)
    plt.plot(x_norm, norm.pdf(x_norm), 'k-', lw=2, alpha=1)
    plt.xlim([normal_samples[0], -normal_samples[0]])
    plt.xlabel('z', fontsize=20)
    plt.ylabel('p(z)', fontsize=20)
    plt.title('probability density function (PDF) of N(0,1)', fontsize=15)

    plt.tight_layout()