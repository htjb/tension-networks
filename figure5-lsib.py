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
from scipy.stats import ecdf
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

base_dir = 'chains/lsbi-different-priors/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Parameters & priors
n = 3
mu = np.random.rand(n)
Sigmas = [0.01, 1, 100]
#fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.3))
fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.3))

for i, Sigma in enumerate(Sigmas):
    print('Iteration ', i, ' Sigma = ', Sigma)

    theta_true = multivariate_normal(mu, Sigma).rvs()

    # Data A
    d = 50
    M = np.random.rand(d, n)
    m = np.random.rand(d)
    C = 0.01
    model_A = LinearModel(M=M, m=m, C=C, 
                          mu=mu, Sigma=Sigma)

    # Data B
    d =  50
    M = np.random.rand(d, n)
    m = np.random.rand(d)
    C = 0.01
    model_B = LinearModel(M=M, m=m, C=C, 
                          mu=mu, Sigma=Sigma)

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

    N_sim = 500000

    AB_sim = model_AB.evidence().rvs(N_sim)
    A_sim = AB_sim[:, :model_A.d]
    B_sim = AB_sim[:, model_A.d:]

    nrei = nre(lr=1e-4)
    nrei.build_model(len(A_obs) + len(B_obs),
                        [25]*5, 'sigmoid')
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

    N_test_sim = 5000
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

    mask = np.isfinite(r)

    Robs = logR(A_obs, B_obs)

    axes[i, 0].hist(r[mask], bins=50, density=True)
    axes[i, 0].axvline(Robs, color='r', ls='--')
    axes[i, 0].set_title(r'$\log R_{obs}=$' + str(np.round(Robs, 2)))

    if i > 0:
        axes[i, 0].set_xlim(axes[0, 0].get_xlim()[0], 
                            axes[0, 0].get_xlim()[1])

    r  = np.sort(r[mask])
    c = ecdf(r)


    sigmaD, sigma_D_upper, sigma_D_lower, \
            sigmaA, sigma_A_upper, sigma_A_lower, \
                sigmaR, sigmaR_upper, sigmaR_lower = \
                    calcualte_stats(Robs, errorRs, c)
    print('Prior Set:', prior_sets_names[i])
    print(f'Rs: {Robs}, Rs_upper: {Robs + errorRs},' + 
            f'Rs_lower: {Robs - errorRs}')
    print(f'sigmaD: {sigmaD}, sigma_D_upper: ' + 
            f'{np.abs(sigmaD - sigma_D_upper)}, ' +
            f'sigma_D_lower: {np.abs(sigma_D_lower - sigmaD)}')
    print(f'sigmaA: {sigmaA}, sigma_A_upper: ' +
            f'{np.abs(sigmaA - sigma_A_upper)}, ' +
            f'sigma_A_lower: {np.abs(sigma_A_lower - sigmaA)}')
    print(f'sigmaR: {sigmaR}, sigmaR_upper: ' + 
            f'{np.abs(sigmaR - sigmaR_upper)}, ' +
            f'sigmaR_lower: {np.abs(sigmaR_lower - sigmaR)}')
    np.savetxt(sbase + f'tension_stats.txt',
                np.hstack([sigmaD, sigma_D_upper, sigma_D_lower,
                            sigmaA, sigma_A_upper, sigma_A_lower,
                            sigmaR, sigmaR_upper, sigmaR_lower]).T)
    
    axes[i, 1].plot(r, c.cdf.evaluate(r))
    axes[i, 1].axhline(c.cdf.evaluate(Robs), ls='--',
                color='r')
    axes[i, 1].set_title(r'$\sigma_D =$' + f'{sigmaD:.3f}' + 
                         r'$+$' + f'{np.abs(sigmaD - sigma_D_upper):.3f}' +
                r'$(-$' + f'{np.abs(sigma_D_lower - sigmaD):.3f}' + r'$)$' + '\n' +
                r'$\sigma_A=$' + f'{sigmaA:.3f}' + 
                r'$+$' + f'{np.abs(sigmaA - sigma_A_upper):.3f}' +
                r'$(-$' + f'{np.abs(sigma_A_lower - sigmaA):.3f}' + r'$)$')
    axes[i, 1].axhspan(c.cdf.evaluate(Robs - errorRs), 
            c.cdf.evaluate(Robs + errorRs), 
            alpha=0.1, 
            color='r')

    if i == 0:
        prior_label = 'Wide'
        axes[i, 0].set_ylabel('Wide Prior\nDensity')
        axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')
    elif i == 1:
        prior_label = 'Conservative'
        axes[i, 0].set_ylabel('Conservative Prior\nDensity')
        axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')
    else:
        prior_label = 'Narrow'
        axes[i, 0].set_ylabel('Narrow Prior\nDensity')
        axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')
        axes[i, 0].set_xlabel(r'$\log R$')
        axes[i, 1].set_xlabel(r'$\log R$')

    
    axes[i, 2].axis('off')
    axes[i, 2].table(cellText=[[str(ps[0, 0]) + ' - ' + str(ps[0, 1])],
                                [str(ps[1, 0]) + ' - ' + str(ps[1, 1])], 
                                [str(ps[2, 0]) + ' - ' + str(ps[2, 1])],
                                [str(ps[3, 0]) + ' - ' + str(ps[3, 1])]],
                     colLabels=[prior_label],
                     rowLabels=[r'$A$', r'$\nu_0$', r'$w$', r'$\sigma$'],
                     cellLoc='center',
                     loc='center',
                     fontsize=15)

plt.tight_layout()
plt.savefig('figures/figure5.pdf', bbox_inches='tight')
plt.close()