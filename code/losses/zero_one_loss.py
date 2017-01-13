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
import loss

class ZeroOneLoss(loss.Loss):
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
        error_rate = np.mean(np.argmax(ypred, 1) != np.argmax(ytrue, 1))
        return error_rate

    def get_name(self):
        return "Error Rate"
