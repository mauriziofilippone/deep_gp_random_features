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
##
## Original code by Karl Krauth 
## Changes by Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone

import tensorflow as tf
import likelihood
import utils

class Softmax(likelihood.Likelihood):
    """
    Implements softmax likelihood for multi-class classification
    """
#    def __init__(self):

    def log_cond_prob(self, output, latent_val):
        return tf.reduce_sum(output * latent_val, 2) - utils.logsumexp(latent_val, 2)

    def predict(self, latent_val):
        """
        return the probabilty for all the samples, datapoints and calsses
        :param latent_val:
        :return:
        """
        logprob = latent_val - tf.expand_dims(utils.logsumexp(latent_val, 2), 2)
        return tf.exp(logprob)

    def get_params(self):
        return None
