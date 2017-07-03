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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes

import gzip

from dataset import DataSet
import utils
import likelihoods
from dgp_rff import DgpRff

# import baselines

import losses

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images_2(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D unit8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = np.int64(_read32(bytestream))
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data



def process_mnist(images, dtype = dtypes.float32, reshape=True):
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images

def get_data_info(images):
    rows, cols = images.shape
    std = np.zeros(cols)
    mean = np.zeros(cols)
    for col in range(cols):
        std[col] = np.std(images[:,col])
        mean[col] = np.mean(images[:,col])
    return mean, std

def standardize_data(images, means, stds):
    data = images.copy()
    rows, cols = data.shape
    for col in range(cols):
        if stds[col] == 0:
            data[:,col] = (data[:,col] - means[col])
        else:
            data[:,col] = (data[:,col] - means[col]) / stds[col]
    return data

def import_mnist():
    """
    This import mnist and saves the data as an object of our DataSet class
    :return:
    """
    VALIDATION_SIZE = 0
    ONE_HOT = True
    TRAIN_DIR = 'INFMNIST_data/'


    train_images = extract_images_2(open(TRAIN_DIR + 'mnist8m-patterns-idx3-ubyte.gz'))

    train_labels = extract_labels(open(TRAIN_DIR + 'mnist8m-labels-idx1-ubyte.gz'), one_hot=ONE_HOT)

    test_images = extract_images(open(TRAIN_DIR + 'test10k-patterns.gz'))

    test_labels = extract_labels(open(TRAIN_DIR + 'test10k-labels.gz'), one_hot=ONE_HOT)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    ## Process images
    train_images = process_mnist(train_images)
    validation_images = process_mnist(validation_images)
    test_images = process_mnist(test_images)

    ## Standardize data
    train_mean, train_std = get_data_info(train_images)
#    train_images = standardize_data(train_images, train_mean, train_std)
#    validation_images = standardize_data(validation_images, train_mean, train_std)
#    test_images = standardize_data(test_images, train_mean, train_std)

    data = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)
    val = DataSet(validation_images, validation_labels)

    return data, test, val


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test, _ = import_mnist()

    ## Here we define a custom loss for dgp to show
    error_rate = losses.ZeroOneLoss(data.Dout)

    ## Likelihood
    like = likelihoods.Softmax()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    dgp = DgpRff(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)

    ## Learning
    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                 FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints)
