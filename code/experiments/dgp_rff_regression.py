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

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

from dataset import DataSet
import utils
import likelihoods
from dgp_rff import DgpRff
import tensorflow as tf
import numpy as np
import losses

def import_dataset(dataset, fold):

    train_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtrain__FOLD_' + fold, delimiter=' ')
    train_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytrain__FOLD_' + fold, delimiter=' ')
    train_Y = np.reshape(train_Y, (-1, 1))
    test_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtest__FOLD_' + fold, delimiter=' ')
    test_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytest__FOLD_' + fold, delimiter=' ')
    test_Y = np.reshape(test_Y, (-1, 1))

    data = DataSet(train_X, train_Y)
    test = DataSet(test_X, test_Y)

    return data, test


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test = import_dataset(FLAGS.dataset, FLAGS.fold)

    ## Here we define a custom loss for dgp to show
    error_rate = losses.RootMeanSqError(data.Dout)

    ## Likelihood
    like = likelihoods.Gaussian()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    dgp = DgpRff(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)

    ## Learning
    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                 FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints)
