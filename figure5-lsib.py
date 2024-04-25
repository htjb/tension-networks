from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensionnet.tensionnet import nre
from tensionnet.utils import coverage_test
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from random import shuffle
from scipy.stats import ecdf, norm
from tqdm import tqdm
import os
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

def logR(A, B):
        return model_AB.evidence().logpdf(np.hstack([A, B])) - \
            model_A.evidence().logpdf(A) - model_B.evidence().logpdf(B)

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


# Parameters & priors
n = 3
# Data B
d =  50
MB = np.random.rand(d, n)
mB = np.random.rand(d)
CB = 0.01
# Data A
MA = np.random.rand(d, n)
mA = np.random.rand(d)
CA = 0.01

mu = np.random.rand(n)

theta_true = multivariate_normal(mu, 0.01).rvs()
#theta_true = [0.70325735, 0.56504433, 0.43477517]
print(theta_true)
"""for i in range(theta_true.shape[0]):
     plt.hist(theta_true[i, :], bins=50, density=True, histtype='step')
plt.show()
sys.exit(1)"""
#theta_true = [0.8, 0.9, 0.95]

Sigmas = [0.1, 1, 100]
prior_label = [r'$\sigma =$' + f'{Sigmas[0]}', 
               r'$\sigma =$' + f'{Sigmas[1]}', 
               r'$\sigma =$' + f'{Sigmas[2]}']
#fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.3))
fig, axes = plt.subplots(3, 3, figsize=(8, 6.3), sharex='col')

for i, Sigma in enumerate(Sigmas):
    print('Iteration ', i, ' Sigma = ', Sigma)

    model_A = LinearModel(M=MA, m=mA, C=CA, 
                          mu=mu, Sigma=Sigma)
    model_B = LinearModel(M=MB, m=mB, C=CB, 
                          mu=mu, Sigma=Sigma)

    # Data AB
    d = model_A.d + model_B.d
    M = np.vstack([model_A.M, model_B.M])
    m = np.hstack([model_A.m, model_B.m])
    C = np.concatenate([model_A.C * np.ones(model_A.d), 
                        model_B.C * np.ones(model_B.d)])
    model_AB = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

    if i == 0:
        A_obs = model_A.likelihood(theta_true).rvs()
        B_obs = model_B.likelihood(theta_true).rvs()

    N_sim = 500000

    AB_sim = model_AB.evidence().rvs(N_sim)
    A_sim = AB_sim[:, :model_A.d]
    B_sim = AB_sim[:, model_A.d:]

    nrei = nre(lr=1e-4)
    nrei.build_model(len(A_obs) + len(B_obs),
                        [25]*5, 'sigmoid')
    norm_data_train, norm_data_test, data_train, data_test, \
        labels_train, labels_test = \
            simulation_process(A_sim, B_sim)
    nrei.data_test = norm_data_test
    nrei.labels_test = labels_test
    nrei.data_train = norm_data_train
    nrei.labels_train = labels_train
    nrei.simulation_func_A = None
    nrei.simulation_func_B = None


    N_test_sim = 5000
    AB_sim = model_AB.evidence().rvs(N_test_sim)
    A_sim = AB_sim[:, :model_A.d]
    B_sim = AB_sim[:, model_A.d:]
    logr_true_dist = logR(A_sim, B_sim)

    #if i ==0:
    Robs = logR(A_obs, B_obs)

    logr_true_dist = np.sort(logr_true_dist)
    true_cdf = ecdf(logr_true_dist)
    true_sigmaD = norm.isf(true_cdf.cdf.evaluate(Robs)/2)
    true_sigmaA = norm.isf((1- true_cdf.cdf.evaluate(Robs))/2)
    print(f'True sigmaD: {true_sigmaD}')
    print(f'True sigmaA: {true_sigmaA}')

    model, data_test, labels_test = nrei.training(epochs=1000,
                                                  batch_size=1000)

    A_sim = (A_sim - data_train[:, :len(A_obs)].mean(axis=0)) / \
        data_train[:, :len(A_obs)].std(axis=0)
    B_sim = (B_sim - data_train[:, len(A_obs):].mean(axis=0)) / \
        data_train[:, len(A_obs):].std(axis=0)

    data_test = np.hstack([A_sim, B_sim])

    nrei.__call__(iters=data_test)
    predicted_r_dist = nrei.r_values

    mask = np.isfinite(predicted_r_dist)

    axes[i, 0].hist(predicted_r_dist[mask], bins=50, density=True, 
                    histtype='step', label='Prediction')
    axes[i, 0].hist(logr_true_dist, bins=50, density=True,
                    histtype='step', label='Truth')

    axes[i, 0].axvline(Robs, color='r', ls='--')
    axes[i, 0].set_title(r'$\log R_{obs}=$' + str(np.round(Robs, 2)))

    predicted_r_dist  = np.sort(predicted_r_dist[mask])
    c = ecdf(predicted_r_dist)

    sigmaD = norm.isf(c.cdf.evaluate(Robs)/2)
    sigmaA = norm.isf((1- c.cdf.evaluate(Robs))/2)
    print(f'Predicted sigmaD: {sigmaD}')
    print(f'Predicted sigmaA: {sigmaA}')

    
    axes[i, 1].plot(predicted_r_dist, c.cdf.evaluate(predicted_r_dist), label='Prediction')
    axes[i, 1].plot(logr_true_dist, true_cdf.cdf.evaluate(logr_true_dist), label='Truth')
    axes[i, 1].axhline(c.cdf.evaluate(Robs), ls='--',
                color='C0')
    axes[i, 1].axhline(true_cdf.cdf.evaluate(Robs), ls='--',
                color='C1')
    
    if i == 0:
         axes[i, 0].legend(fontsize=8)
         axes[i, 1].legend(fontsize=8)
        
    [axes[i, j].tick_params(labelbottom=True) for j in range(2)]

    axes[i, 0].set_ylabel(prior_label[i])
    axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')

    axes[i, 2].axis('off')
    axes[i, 2].table(cellText=[[f'{true_sigmaD:.3f}',
                                f'{true_sigmaA:.3f}'],
                                [f'{sigmaD:.3f}',
                                 f'{sigmaA:.3f}']],
                     colLabels=[r'$\sigma_D$', r'$\sigma_A$'],
                     rowLabels=['Truth', 'Prediction'],
                     cellLoc='center',
                     loc='center',
                     fontsize=15)

plt.tight_layout()
plt.savefig('figures/figure5-lsbi.pdf', bbox_inches='tight')
plt.close()