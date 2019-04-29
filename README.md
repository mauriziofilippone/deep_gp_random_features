# README #

This repository contains code to reproduce the results in the paper:

K. Cutajar, E. V. Bonilla, P. Michiardi, and M. Filippone. Random feature expansions for deep Gaussian processes. In Proceedings of the 33rd International Conference on Machine Learning, ICML 2017, Sydney, Australia, August 6-11, 2017, 2017.
(Link: http://proceedings.mlr.press/v70/cutajar17a/cutajar17a.pdf)

The code is written in Python and uses the TensorFlow module; follow https://www.tensorflow.org to install TensorFlow.

Currently the code is structured so that the learning of DGPs is done using stochastic gradient optimization and a loss function of interest is displayed on the test error every fixed number of iterations.

29/04/2019 - The original implementation has been updated to work with Python 3.6 and the latest version of TensorFlow (1.13.1). This latest version of the code also introduces an additional option for utilising the local reparameterization trick in order to accelerate convergence during training.

## Flags ##

The code implements variational inference for a deep Gaussian process approximated using random Fourier features. The code accepts the following options:

*   -h, --help            Show help message and exit
*   --batch_size          Batch size
*   --learning_rate       Initial learning rate
*   --n_iterations        Number of iterations (batches) to train the DGP model
*   --display_step        Display progress every FLAGS.display_step iterations
*   --mc_train            Number of Monte Carlo samples used to compute stochastic gradients
*   --mc_test             Number of Monte Carlo samples for predictions
*   --n_rff               Number of random features for each layer
*   --df                  Number of GPs per hidden layer
*   --nl                  Number of layers
*   --optimizer           Select the optimizer: it can be adam, adagrad, adadelta, sgd
*   --kernel_type         Kernel: it can be RBF or arccosine
*   --kernel_arccosine_degree  Degree parameter of arc-cosine kernel
*   --is_ard              Using ARD kernel or isotropic
*   --feed_forward        Feed original inputs to each layer
*   --local_reparam       Use the local reparameterization trick
*   --q_Omega_fixed       Number of iterations to keep posterior over Omega fixed
*   --theta_fixed         Number of iterations to keep theta fixed
*   --learn_Omega         How to treat Omega - it can be 'prior_fixed' for Omega obtained from the prior with fixed randomness, 'var_fixed' for variational with fixed randomness, or 'var_resampled' for variational with resampling
*   --duration            Duration of job in minutes
*   --dataset             Dataset name
*   --fold                Dataset fold
*   --seed                Seed for random tf and np operations
*   --less_prints         Disables evaluations involving on the complete training set

Flags for the distributed version

*   --ps_hosts            Comma-separated list of hostname:port pairs
*   --worker_hosts        Comma-separated list of hostname:port pairs
*   --job_name            One of 'ps', 'worker'
*   --task_index          Index of task within the job


## Examples ##

Here are a few examples to run the Deep GP model on various datasets (we assume that the code directory is in PYTHONPATH - otherwise, please append PYTHONPATH=. at the beginning of the commands below):

### Regression ###

```
#!bash
# Learn a DGP model with two layers (nl=2) and three GPs in the hidden layer (df=3) on a regression problem (dataset=concrete). The kernel is RBF by default and use the ARD formulation. Approximate the GPs with 100 random Fourier features.
# Set the optimizer to adam with step-size of 0.01 with a batch size of 200. Use 100 Monte Carlo samples to estimate stochastic gradients (mc_train=100) and use 100 Monte Carlo samples to carry out predictions (mc_test=100).
# Cap the running of the code to 60min and to 100K iterations. Learn Omega variationally, and fix the approximate posterior over Omega and the GP covariance parameters for the first 1000 and 4000 iterations, respectively (q_Omega_fixed=1000 and theta_fixed=4000).

python experiments/dgp_rff_regression.py --seed=12345 --dataset=concrete --fold=1 --q_Omega_fixed=1000 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=var_resampled

```

```
#!bash
# Here is an example where we fix the random Fourier features from the prior induced by the GP approximation (learn_Omega=prior_fixed).

python experiments/dgp_rff_regression.py --seed=12345 --dataset=concrete --fold=1 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=prior_fixed --local_reparam=True

```

### Binary Classification ###
```
#!bash
# Same as the first example but for a classification problem (dataset=credit) and optimizing the spectral frequencies Omega (learn_Omega=var_fixed).  

python experiments/dgp_rff_classification.py --seed=12345 --dataset=credit --fold=1 --q_Omega_fixed=1000 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=var_fixed --local_reparam=True

```

```
#!bash
# Here is an example with the arc-cosine kernel of degree 1.  

python experiments/dgp_rff_classification.py --seed=12345 --dataset=credit --fold=1 --q_Omega_fixed=0 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.01 --n_rff=100 --df=3 --batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 --learn_Omega=var_fixed --kernel_type=arccosine --kernel_arccosine_degree=1 --local_reparam=True

```

### MNIST (Multiclass classification) ###

```
#!bash
# Here is the MNIST example, where we use a two-layer DGP with 50 GPs in the hidden layer. We use 500 random Fourier features to approximate the GPs.
# In this example we use the option less_prints to avoid computing the loss on the full training data every 250 iterations.

python experiments/dgp_rff_mnist.py --seed=12345 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.001 --n_rff=500 --df=50 --batch_size=1000 --mc_train=100 --mc_test=50 --n_iterations=100000 --display_step=250 --less_prints=True --duration=1200 --learn_Omega=var_fixed --local_reparam=True

```

### MNIST8M (Multiclass classification) ###

```
#!bash
# Here is the MNIST8M example - same settings as the MNIST example
# NOTE: Before running the code, please download the infinite MNIST dataset from here: http://leon.bottou.org/_media/projects/infimnist.tar.gz

python experiments/dgp_rff_infmnist.py --seed=12345 --theta_fixed=4000 --is_ard=True --optimizer=adam --nl=2 --learning_rate=0.001 --n_rff=500 --df=50 --batch_size=1000 --mc_train=40 --mc_test=100 --n_iterations=100000 --display_step=1000 --less_prints=True --duration=1200 --learn_Omega=var_fixed

```
