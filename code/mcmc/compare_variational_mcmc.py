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

## This experiment compares the variational approximation with an MCMC sampler for a two-layer DGP model

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

from dataset import DataSet
import utils
import likelihoods
from dgp_rff_remote import DgpRff
import tensorflow as tf
import numpy as np
import losses

from mcmc import MCMC

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Define a function to generate data f(x) as a composition of h(x) with itself
def h(x):
    return np.exp(-np.square(x)) * (2.0 * x)

def f(x):
    return h(h(x))


## Generate some noisy data around f(x)
def generate_toy_data():

    N = 50
    DATA_X = np.random.uniform(-5.0, 5.0, [N, 1])

    true_log_lambda = -2.0
    true_std = np.exp(true_log_lambda) / 2.0  # 0.1
    DATA_y = f(DATA_X) + np.random.normal(0.0, true_std, [N, 1])

    Xtest = np.asarray(np.arange(-10.0, 10.0, 0.1))
    Xtest = Xtest[:, np.newaxis]
    ytest = f(Xtest) # + np.random.normal(0, true_std, [Xtest.shape[0], 1])

    data = DataSet(DATA_X, DATA_y)
    test = DataSet(Xtest, ytest, shuffle=False)

    return data, test


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test = generate_toy_data()

    ## Here we define a custom loss for dgp to show
    error_rate = losses.RootMeanSqError(data.Dout)

    ## Likelihood
    like = likelihoods.Gaussian()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    dgp = DgpRff(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)

    ## Learning
    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_epochs, optimizer,
                 FLAGS.display_step, test, FLAGS.mc_test, error_rate)

    ## Save various quantities of interest so that these can be plotted later
    layers = dgp.session.run(dgp.layer, feed_dict={dgp.X:test.X, dgp.Y:test.Y, dgp.mc:FLAGS.mc_test})
    predictions_variational_F1 = np.zeros([test.Y.shape[0],FLAGS.mc_test])
    predictions_variational_F2 = np.zeros([test.Y.shape[0],FLAGS.mc_test])
    for i in range(FLAGS.mc_test):
        predictions_variational_F1[:,i] = layers[1][i,:,0]
        predictions_variational_F2[:,i] = layers[2][i,:,0]

    np.savetxt("./mcmc/predictions_variational_F1.txt", predictions_variational_F1, fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/predictions_variational_F2.txt", predictions_variational_F2, fmt="%f", delimiter='\t')

    np.savetxt("./mcmc/X.txt", data.X, fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/Xtest.txt", test.X, fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/Y.txt", data.Y, fmt="%f", delimiter='\t')

    np.savetxt("./mcmc/log_theta_sigma2.txt", dgp.session.run(dgp.log_theta_sigma2), fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/log_theta_lengthscale.txt", dgp.session.run(dgp.log_theta_lengthscale), fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/log_lambda.txt", dgp.session.run([dgp.likelihood.log_var]), fmt="%f", delimiter='\t')
    
    ## Run the MCMC sampler and save some quantities of interest for comparison with the variational approximation
    samples_F1, samples_F2, predictions_F1, predictions_F2 = MCMC(data.X, data.Y, test.X)
    np.savetxt("./mcmc/predictions_MCMC_F1.txt", predictions_F1, fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/predictions_MCMC_F2.txt", predictions_F2, fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/samples_MCMC_F1.txt", samples_F1, fmt="%f", delimiter='\t')
    np.savetxt("./mcmc/samples_MCMC_F2.txt", samples_F2, fmt="%f", delimiter='\t')


    ## Here are some plots just to check what is going on - a neater plotting function is in a separate file
    ## Layer 1
    plt.clf()
    plt.plot(test.X, h(test.X), 'b-', linewidth=1, label="True")

    plt.plot(test.X, predictions_variational_F1, 'g-', alpha=0.01, linewidth=10.0)
    plt.plot(test.X, predictions_F1, 'r-', alpha=0.01, linewidth=10.0)
    plt.ylim([-6, 6])
    plt.savefig('./mcmc/figure-posterior-layer1.png')

    ## Layer 2
    plt.clf()
    plt.plot(data.X, data.Y, 'bo', label="Data")
    plt.plot(test.X, test.Y, 'b-', linewidth=1, label="True")

    plt.plot(test.X, predictions_variational_F2, 'g-', alpha=0.01, linewidth=10.0)
    plt.plot(test.X, predictions_F2, 'r-', alpha=0.01, linewidth=10.0)
    plt.ylim([-1.3, 1.3])
    plt.savefig('./mcmc/figure-posterior-layer2.png')
