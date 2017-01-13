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

import abc

class Likelihood:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_cond_prob(self, output, latent_val):
        """
        Subclass should implement log p(Y | F)
        :param output:  (batch_size x Dout) matrix containing true outputs
        :param latent_val: (MC x batch_size x Q) matrix of latent function values, usually Q=F
        :return:
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def predict(self, latent_val):
        raise NotImplementedError("Subclass should implement this.")

