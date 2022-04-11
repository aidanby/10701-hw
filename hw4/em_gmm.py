# -*- coding: utf-8 -*-
"""em gmm

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1obBbFzfCD8VcLw6CAhV6gcgXunc5VqGp
"""

import numpy as np
import matplotlib.pyplot as plt

"""## Multivariate Gaussian function

$f(\mathbf{x}, \boldsymbol{\mu}, \Sigma) = (2\pi)^{-M/2}|\Sigma|^{-1/2}~e^{\frac{-1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$

"""

def gaussian(x, mu, cov):
    # x and mu should be vectors in numpy, shape=(2,)
    # cov should be a matrix in numpy, shape=(2,2)
    M = 2
    scale = (2*np.pi)**(-M/2)*np.linalg.det(cov)**(-1/2)
    return scale*np.exp(-(1/2)*(x-mu).T @ np.linalg.inv(cov) @ (x-mu))

"""## Plot Gaussian contours function"""

def plot_gaussian(mu, cov, x1_min=-10, x1_max=10, x2_min=-10, x2_max=10, color=None):
    x1_values = np.linspace(x1_min, x1_max, 101)
    x2_values = np.linspace(x2_min, x2_max, 101)

    x1_grid, x2_grid = np.meshgrid(x1_values,x2_values)

    M,N = x1_grid.shape
    y_grid = np.zeros((M,N))

    x = np.zeros((2,))

    for i in range(M):
        for j in range(N):
            x[0] = x1_grid[i,j]
            x[1] = x2_grid[i,j]

            y_grid[i,j] = gaussian(x, mu, cov)

    plt.contour(x1_grid, x2_grid, y_grid, colors=color)

"""## Load data
Note: The code assumes that the data file is in the same folder as the jupyter notebook. In Google colab, you can upload the file directly into the workspace by in the Files tab on the left.
"""

X = np.loadtxt("./gmm_data.csv", delimiter=",")
print(X.shape)

"""## Initial parameters"""

K=4

mu_list = []
sigma_list = []
pi_list = []

for k in range(K):
    mu_list.append((k+1)*np.ones((2,)))

    sigma_list.append(np.eye(2))

    pi_list.append(1/K)

"""## Plot data with initial Gaussian contours"""

# Square figure size
plt.figure(figsize=(8,8))

# Plot points
plt.plot(X[:,0], X[:,1], 'o', markerfacecolor="None", alpha=0.3)

# Plot K Gaussians
colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for k in range(K):
    plot_gaussian(mu_list[k], sigma_list[k], color=colors[k])

# Axes
plt.gca().axhline(y=0, color='gray')
plt.gca().axvline(x=0, color='gray')

# Labels
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("$x_2$", fontsize=20)
plt.title("Iteration 0", fontsize=20)

def compute_responsibilities(X, pi_list, mu_list, sigma_list):
  '''E-step: compute responsibilities, given the current parameters'''
  num_data = len(X)
  resp = np.zeros((num_data, K))
  # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
  # Hint: To compute likelihood of seeing data point i given cluster k, use multivariate_normal.pdf.
  for i in range(num_data):
      for j in range(K):
          resp[i][j] = pi_list[j]*gaussian(X[i], mu_list[j], sigma_list[j])

  # Add up responsibilities over each data point and normalize
  row_sums = resp.sum(axis=1)[:, np.newaxis]
  resp = resp / row_sums
  
  return resp

# M-step
def compute_soft_counts(resp):
    # Compute the total responsibility assigned to each cluster, which will be useful when 
    # implementing M-steps below. In the lectures this is called N^{soft}
    counts = np.sum(resp, axis=0)
    return counts

def compute_weights(counts):
    num_clusters = len(counts)
    weights = [0.] * num_clusters
    
    for k in range(num_clusters):
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        # HINT: compute # of data points by summing soft counts.
        # YOUR CODE HERE
        weights[k] = counts[k] / np.sum(counts)

    return weights

def compute_means(data, resp, counts):
    num_clusters = len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * num_clusters
    
    for k in range(num_clusters):
        # Update means for cluster k using the M-step update rule for the mean variables.
        # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        weighted_sum = 0.
        for i in range(num_data):
            # YOUR CODE HERE
            weighted_sum += data[i] * resp[i][k]
        # YOUR CODE HERE
        means[k] = weighted_sum / counts[k]

    return means

def compute_covariances(data, resp, counts, means):
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim,num_dim))] * num_clusters
    
    for k in range(num_clusters):
        # Update covariances for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        weighted_sum = np.zeros((num_dim, num_dim))
        for i in range(num_data):
            # YOUR CODE HERE (Hint: Use np.outer on the data[i] and this cluster's mean)
            weighted_sum += resp[i][k]*np.outer(data[i] - means[k], data[i] - means[k])
        # YOUR CODE HERE
        covariances[k] = weighted_sum / counts[k]
    return covariances

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll

def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for it in range(maxiter):
        print("Iteration %s" % it)

        resp = compute_responsibilities(data, weights, means, covariances)

        counts = compute_soft_counts(resp)

        weights = compute_weights(counts)
      
        means = compute_means(data, resp, counts)
        
        covariances = compute_covariances(data, resp, counts, means)
        
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}
    return out

results = EM(X, mu_list, sigma_list, pi_list, maxiter=1)
# Square figure size
plt.figure(figsize=(8,8))

plt.plot(X[:,0], X[:,1], 'o', markerfacecolor="None", alpha=0.3)
colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for i in range(K):
  plot_gaussian(results['means'][i], results['covs'][i])

# Axes
plt.gca().axhline(y=0, color='gray')
plt.gca().axvline(x=0, color='gray')

# Labels
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("$x_2$", fontsize=20)
plt.title("Iteration 100", fontsize=20)