## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## These functions implement an MCMC sampler for a DGP model with two layers.
## Unlike the rest of the repository, the implementation is in python (no TensorFlow) and it is not optimized - it is just to qualitatively assess the accuracy of the variational approximation.

# PYTHONPATH=.  python mcmc/compare_variational_mcmc.py --Omega_fixed=True --theta_fixed=False --n_iter_fixed=25001 --seed=98132 --optimizer=adam --nl=2 --learning_rate=0.001 --n_rff=10 --df=1 --mc_train=20 --batch_size=50 --mc_test=100 --n_epochs=25001 --display_step=1000
# PYTHONPATH=.  python mcmc/compare_variational_mcmc.py --Omega_fixed=True --theta_fixed=False --n_iter_fixed=25001 --seed=98132 --optimizer=adam --nl=2 --learning_rate=0.001 --n_rff=50 --df=1 --mc_train=20 --batch_size=50 --mc_test=100 --n_epochs=25001 --display_step=1000

from __future__ import print_function

import numpy as np
from scipy import array, linalg, dot


## Define the covariance function (RBF and isotropic)
def covariance_function(x1, x2, log_theta_sigma2, log_theta_lengthscale):

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    
    K = np.zeros([n1, n2])

    ## The kernel matrix can be computed more efficiently by avoiding this loop
    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.exp(log_theta_sigma2) * np.exp(- 0.5 * np.dot(x1[i,:] - x2[j,:], x1[i,:] - x2[j,:]) / np.exp(2.0 * log_theta_lengthscale))

    return K


## This implements a Gaussian likelihood
def log_p_Y_given_F1(Y, F1, log_theta):

    log_theta_sigma2, log_theta_lengthscale, log_theta_lambda = unpack_log_theta(log_theta)

    n = Y.shape[0]

    K_Y = covariance_function(F1, F1, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(n) * np.exp(log_theta_lambda)
    L_K_Y, lower_K_Y = linalg.cho_factor(K_Y, lower=True)

    nu = linalg.solve_triangular(L_K_Y, Y, lower=True)

    return -np.sum(np.log(np.diagonal(L_K_Y))) - 0.5 * np.dot(nu.transpose(), nu)


## Elliptical Slice Sampling to sample from the posterior over latent variables at layer 1 (the latent variables at layer 2 are integrated out analytically)
def do_sampleF1(Y, X, current_F1, log_theta):

    log_theta_sigma2, log_theta_lengthscale, log_theta_lambda = unpack_log_theta(log_theta)
    
    n = Y.shape[0]

    current_logy = log_p_Y_given_F1(Y, current_F1, log_theta) + np.log(np.random.uniform(0.0, 1.0, 1))

    proposed_F1 = current_F1 * 1.0

    K_F1 = covariance_function(X, X, log_theta_sigma2[0], log_theta_lengthscale[0]) + np.eye(n) * 1e-9
    L_K_F1 = linalg.cholesky(K_F1, lower=True)
    auxiliary_nu = np.dot(L_K_F1, np.random.normal(0.0, 1.0, [n,1]))

    auxiliary_theta = np.random.uniform(0.0, 2 * np.pi, 1)
    auxiliary_thetamin = auxiliary_theta - 2 * np.pi
    auxiliary_thetamax = auxiliary_theta * 1.0

    while True:
        proposed_F1 = current_F1 * np.cos(auxiliary_theta) + auxiliary_nu * np.sin(auxiliary_theta)
        proposed_logy = log_p_Y_given_F1(Y, proposed_F1, log_theta)

        if proposed_logy > current_logy:
            break
    
        if(auxiliary_theta < 0):
            auxiliary_thetamin = auxiliary_theta * 1.0
        if(auxiliary_theta >= 0):
            auxiliary_thetamax = auxiliary_theta * 1.0
        
        auxiliary_theta = np.random.uniform(auxiliary_thetamin, auxiliary_thetamax, 1)

    return proposed_F1

## Function to draw directly from the posterior over latent variables at layer 2
def do_sampleF2(Y, X, current_F1, log_theta):

    log_theta_sigma2, log_theta_lengthscale, log_theta_lambda = unpack_log_theta(log_theta)

    n = Y.shape[0]

    K_F2 = covariance_function(current_F1, current_F1, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(n) * 1e-9
    K_Y = K_F2 + np.eye(n) * np.exp(log_theta_lambda)
    L_K_Y, lower_K_Y = linalg.cho_factor(K_Y, lower=True)
    K_inv_Y = linalg.cho_solve((L_K_Y, lower_K_Y), Y)

    mu = np.dot(K_F2, K_inv_Y)

    K_inv_K = linalg.cho_solve((L_K_Y, lower_K_Y), K_F2)
    Sigma = K_F2 - np.dot(K_F2, K_inv_K)

    L_Sigma = linalg.cholesky(Sigma, lower=True)
    proposed_F2 = mu + np.dot(L_Sigma, np.random.normal(0.0, 1.0, [n, 1]))

    return proposed_F2

## Unpack the vector of parameters into their three elements
def unpack_log_theta(log_theta):
    return log_theta[0], log_theta[1], log_theta[2]

## Main MCMC function
def MCMC(X, Y, Xtest, n_MCMC = 100, nburnin = 10, save_every = 10):
    n = X.shape[0]
    ntest = Xtest.shape[0]

    ## Load the parameters from disk - these have been saved by the main function calling MCMC after the variational approximation
    log_theta_sigma2 = np.loadtxt("./mcmc/log_theta_sigma2.txt", delimiter='\t')
    log_theta_lengthscale = np.loadtxt("./mcmc/log_theta_lengthscale.txt", delimiter='\t')
    log_theta_lambda = np.loadtxt("./mcmc/log_lambda.txt", delimiter='\t')
    
    ## Pack the parameters into a vector
    log_theta = (log_theta_sigma2, log_theta_lengthscale, log_theta_lambda)

    ## Initialize the containers of the samples and the predictions on test data in the two layers
    samples_F1 = np.zeros([n, n_MCMC])
    samples_F2 = np.zeros([n, n_MCMC])

    predictions_F1 = np.zeros([ntest, n_MCMC])
    predictions_F2 = np.zeros([ntest, n_MCMC])

    ## Initialize the MCMC sampler
    current_F1 = np.zeros([n, 1])
    current_F2 = np.zeros([n, 1])

    ## Main MCMC loop
    for iteration_MCMC in range(-nburnin,n_MCMC):
        
        print("i=", iteration_MCMC)
        
        ## Apply the transition operators "save_every" times
        for inner_iteration_MCMC in range(save_every):
            current_F1 = do_sampleF1(Y, X, current_F1, log_theta)
            current_F2 = do_sampleF2(Y, X, current_F1, log_theta)

        ## Start recording only after burn-in
        if iteration_MCMC >= 0:
            samples_F1[:,iteration_MCMC] = current_F1[:,0]
            samples_F2[:,iteration_MCMC] = current_F2[:,0]

            ## Predict on test data - layer 1
            K_F1 = covariance_function(X, X, log_theta_sigma2[0], log_theta_lengthscale[0]) + np.eye(n) * 1e-9
            L_F1, lower_K_F1 = linalg.cho_factor(K_F1, lower=True)
            K_star = covariance_function(X, Xtest, log_theta_sigma2[0], log_theta_lengthscale[0])
            K_star_star = covariance_function(Xtest, Xtest, log_theta_sigma2[0], log_theta_lengthscale[0]) + np.eye(ntest) * 1e-9
            L_inv_K_star = linalg.solve_triangular(L_F1, K_star, lower=True)
            L_inv_F1 = linalg.solve_triangular(L_F1, current_F1, lower=True)

            mu_star = np.dot(L_inv_K_star.transpose(), L_inv_F1)
            Sigma_star = K_star_star - np.dot(L_inv_K_star.transpose(), L_inv_K_star)
            L_Sigma_star = linalg.cholesky(Sigma_star, lower=True)

            predictions_F1[:,iteration_MCMC] = (np.dot(L_Sigma_star, np.random.normal(0, 1, [ntest, 1])) + mu_star)[:,0]

            ## Predict on test data - layer 2
            K_F2 = covariance_function(current_F1, current_F1, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(n) * 1e-9
            L_K_F2, lower_K_F2 = linalg.cho_factor(K_F2, lower=True)

            TMP = predictions_F1[:,iteration_MCMC]
            TMP = np.reshape(TMP, [ntest,1])
            K_star = covariance_function(current_F1, TMP, log_theta_sigma2[1], log_theta_lengthscale[1])
            K_star_star = covariance_function(TMP, TMP, log_theta_sigma2[1], log_theta_lengthscale[1]) + np.eye(ntest) * 1e-9

            L_inv_K_star = linalg.solve_triangular(L_K_F2, K_star, lower=True)
            L_inv_F2 = linalg.solve_triangular(L_K_F2, current_F2, lower=True)

            mu_star = np.dot(L_inv_K_star.transpose(), L_inv_F2)

            Sigma_star = K_star_star - np.dot(L_inv_K_star.transpose(), L_inv_K_star)
            L_Sigma_star = linalg.cholesky(Sigma_star, lower=True)

            predictions_F2[:,iteration_MCMC] = (np.dot(L_Sigma_star, np.random.normal(0, 1, [ntest, 1])) + mu_star)[:,0]

    return samples_F1, samples_F2, predictions_F1, predictions_F2
