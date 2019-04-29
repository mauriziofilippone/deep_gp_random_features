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
import tensorflow as tf

## Log-density of a univariate Gaussian distribution
def log_norm_pdf(x, m=0.0, log_v=0.0):
    return - 0.5 * tf.log(2 * np.pi) - 0.5 * log_v - 0.5 * tf.square(x - m) / tf.exp(log_v)

## Kullback-Leibler divergence between multivariate Gaussian distributions q and p with diagonal covariance matrices
def DKL_gaussian(mq, log_vq, mp, log_vp):
    """
    KL[q || p]
    :param mq: vector of means for q
    :param log_vq: vector of log-variances for q
    :param mp: vector of means for p
    :param log_vp: vector of log-variances for p
    :return: KL divergence between q and p
    """
    log_vp = tf.reshape(log_vp, (-1, 1))
    return 0.5 * tf.reduce_sum(log_vp - log_vq + (tf.pow(mq - mp, 2) / tf.exp(log_vp)) + tf.exp(log_vq - log_vp) - 1)

## Draw a tensor of standard normals
def get_normal_samples(ns, din, dout):
    """"
    :param ns: Number of samples
    :param din:
    :param dout:
    :return:
    """
    dx = np.amax(din)
    dy = np.amax(dout)
    return tf.random_normal(shape=[ns, dx, dy], dtype="float32")

## Log-sum operation
def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m), dim))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), dim))


## Get flags from the command line
def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 50, 'Batch size.  ')
    flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
    flags.DEFINE_integer('n_iterations', 2000, 'Number of iterations (batches) to feed to the DGP')
    flags.DEFINE_integer('display_step', 100, 'Display progress every FLAGS.display_step iterations')
    flags.DEFINE_integer('mc_train', 30, 'Number of Monte Carlo samples used to compute stochastic gradients')
    flags.DEFINE_integer('mc_test', 30, 'Number of Monte Carlo samples for predictions')
    flags.DEFINE_integer('n_rff', 10, 'Number of random features for each layer')
    flags.DEFINE_integer('df', 1, 'Number of GPs per hidden layer')
    flags.DEFINE_integer('nl', 1, 'Number of layers')
    flags.DEFINE_string('optimizer', "adagrad", 'Optimizer')
    flags.DEFINE_string('kernel_type', "RBF", 'arccosine')
    flags.DEFINE_integer('kernel_arccosine_degree', 1, 'Degree parameter of arc-cosine kernel')
    flags.DEFINE_boolean('is_ard', False, 'Using ARD kernel or isotropic')
    flags.DEFINE_boolean('local_reparam', False, 'Using the local reparameterization trick')
    flags.DEFINE_boolean('feed_forward', False, 'Feed original inputs to each layer')
    flags.DEFINE_integer('q_Omega_fixed', 0, 'Number of iterations to keep posterior over Omega fixed')
    flags.DEFINE_integer('theta_fixed', 0, 'Number of iterations to keep theta fixed')
    flags.DEFINE_string('learn_Omega', 'prior_fixed', 'How to treat Omega - fixed (from the prior), optimized, or learned variationally')
    flags.DEFINE_integer('duration', 10000000, 'Duration of job in minutes')

    # Flags for use in cluster experiments
    tf.app.flags.DEFINE_string("dataset", "", "Dataset name")
    tf.app.flags.DEFINE_string("fold", "1", "Dataset fold")
    tf.app.flags.DEFINE_integer("seed", 0, "Seed for random tf and np operations")
    tf.app.flags.DEFINE_boolean("less_prints", False, "Disables evaluations involving the complete dataset without batching")

    # Flags to setup distributed exectuion
    # Flags for defining the tf.train.ClusterSpec
    # These can be set by using the following CLI arguments:
    # --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
    # --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
    tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
    tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
    # Flags for defining the tf.train.Server
    tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    return FLAGS

## Define the right optimizer for a given flag from command line
def get_optimizer(opt_name, learning_rate):
    switcher = {
        "adagrad": tf.train.AdagradOptimizer(learning_rate),
        "sgd": tf.train.GradientDescentOptimizer(learning_rate),
        "adam": tf.train.AdamOptimizer(learning_rate),
        "adadelta": tf.train.AdadeltaOptimizer(learning_rate)
    }
    return switcher.get(opt_name)
