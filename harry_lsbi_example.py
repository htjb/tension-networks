from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensionnet.tensionnet import nre
from tensionnet.utils import coverage_test
import tensorflow as tf
import numpy as np
from random import shuffle
from tqdm import tqdm
import os

def simulation_process(simsA, simsB):
    # generate lots of simulations 
        
    idx = np.arange(0, len(simsB), 1)
    shuffle(idx)
    mis_labeled_simsB = simsB[idx]

    data = []
    for i in range(len(simsA)):
        """
        Sigma(log(r)) = 1 results in R >> 1 i.e. data sets are consistent
        sigma(log(r)) = 0 --> R << 1 i.e. data sets are inconsistent
        """
        data.append([*simsA[i], *simsB[i], 1]) 
        data.append([*simsA[i], *mis_labeled_simsB[i], 0])
    data = np.array(data)
    idx = np.arange(0, 2*len(simsA), 1)
    shuffle(idx)
    labels = data[idx, -1]
    data = data[idx, :-1]
    
    print('Simulations built.')
    print('Splitting data and normalizing...')

    data_train, data_test, labels_train, labels_test = \
            train_test_split(data, labels, test_size=0.2)
    
    labels_test = labels_test
    labels_train = labels_train

    data_trainA = data_train[:, :len(simsA[0])]
    data_trainB = data_train[:, len(simsA[0]):]
    data_testA = data_test[:, :len(simsA[0])]
    data_testB = data_test[:, len(simsA[0]):]

    data_testA = (data_testA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_testB = (data_testB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)
    data_trainA = (data_trainA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_trainB = (data_trainB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)

    norm_data_train = np.hstack([data_trainA, data_trainB])
    norm_data_test = np.hstack([data_testA, data_testB])
    return norm_data_train, norm_data_test, data_train, \
        data_test, labels_train, labels_test

# General model is 
# D = m + M theta +/- sqrt(C)
# theta = mu +/- sqrt(Sigma)

base_dir = 'validation/'
label = '_test'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Parameters & priors
n = 3
mu = np.random.rand(n)
Sigmas = [0.01, 1, 100]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax = axes.flatten()
for i, Sigma in enumerate(Sigmas):
    print('Iteration ', i, ' Sigma = ', Sigma)

    theta_true = multivariate_normal(mu, Sigma).rvs()

    # Data A
    d = 20
    M = np.random.rand(d, n)
    m = np.random.rand(d)
    C = 0.01
    model_A = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    # Data B
    d = 20
    M = np.random.rand(d, n)
    m = np.random.rand(d)
    C = 0.01
    model_B = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    # Data AB
    d = model_A.d + model_B.d
    M = np.vstack([model_A.M, model_B.M])
    m = np.hstack([model_A.m, model_B.m])
    C = np.concatenate([model_A.C * np.ones(model_A.d), 
                        model_B.C * np.ones(model_B.d)])
    model_AB = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)


    A_obs = model_A.likelihood(theta_true).rvs()
    B_obs = model_B.likelihood(theta_true).rvs()

    def logR(A, B):
        return model_AB.evidence().logpdf(np.hstack([A, B])) - \
            model_A.evidence().logpdf(A) - model_B.evidence().logpdf(B)

    N_sim = 100000

    AB_sim = model_AB.evidence().rvs(N_sim)
    A_sim = AB_sim[:, :model_A.d]
    B_sim = AB_sim[:, model_A.d:]

    nrei = nre(lr=1e-3)
    nrei.build_model(len(A_obs) + len(B_obs),
                        [20]*10, 'tanh')
    norm_data_train, norm_data_test, data_train, data_test, labels_train, labels_test = \
        simulation_process(A_sim, B_sim)
    nrei.data_test = norm_data_test
    nrei.labels_test = labels_test
    nrei.data_train = norm_data_train
    nrei.labels_train = labels_train
    nrei.simulation_func_A = None
    nrei.simulation_func_B = None


    model, data_test, labels_test = nrei.training(epochs=1000, 
                                                    batch_size=1000)

    N_test_sim = 1000
    AB_sim = model_AB.evidence().rvs(N_test_sim)
    A_sim = AB_sim[:, :model_A.d]
    B_sim = AB_sim[:, model_A.d:]
    logr = logR(A_sim, B_sim)

    A_sim = (A_sim - data_train[:, :len(A_obs)].mean(axis=0)) / \
        data_train[:, :len(A_obs)].std(axis=0)
    B_sim = (B_sim - data_train[:, len(A_obs):].mean(axis=0)) / \
        data_train[:, len(A_obs):].std(axis=0)

    data_test = np.hstack([A_sim, B_sim])

    nrei.__call__(iters=data_test)
    r = nrei.r_values

    alpha, cov = coverage_test(A_sim[:100, :], B_sim[:100, :], nrei)

    ax[i+1].plot(1 - alpha, cov, label='ECP')
    ax[i+1].plot(1-alpha, 1-alpha, label='Expected')
    ax[i+1].set_xlabel(r'$1 - \alpha$')
    ax[i+1].set_ylabel('ECP')
    ax[i+1].legend()
    ax[i+1].set_title(r'$\sigma = $' + str(Sigma))
    


    hist, bins = np.histogram(logr, bins=50)
    ax[0].hist(logr, bins=50, histtype='step', ls='-', 
              label=r'True, $\sigma = $' + str(Sigma), color='C' + str(i))

    print('Surviving Simulations: ', len(r))
    ax[0].hist(r, bins=50, histtype='step', ls='--', 
             label=r'Predicted, $\sigma = $' + str(Sigma), color='C' + str(i))

from scipy.special import expit

x = np.linspace(-20, 20, 1000)
ax[0].plot(x, expit(x)*hist.max(), label='Sigmoid')
#axes.axvline(0.75, ls='--', c='k', label='Threshold for\ncorrectly classified')

plt.legend()
plt.xlabel(r'$\log(R)$')
plt.ylabel('Counts')
plt.savefig(base_dir + 'validation' + label + '.png', dpi=300, bbox_inches='tight')
plt.savefig(base_dir + 'validation' + label + '.pdf')
plt.show()
