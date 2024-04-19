import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from matplotlib import rc
from tensionnet.tensionnet import nre
from random import shuffle
from sklearn.model_selection import train_test_split
from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal
from scipy.stats import ks_2samp
import os

np.random.seed(42)
tf.random.set_seed(42)

# plotting stuff
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', 
     '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

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

base = 'parameter-sweep_lsbi_sim_test/'
if not os.path.exists(base):
    os.mkdir(base)

# Parameters & priors
n = 3
mu = np.random.rand(n)
Sigma = 10

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

def logR(A, B):
    return model_AB.evidence().logpdf(np.hstack([A, B])) - \
        model_A.evidence().logpdf(A) - model_B.evidence().logpdf(B)

N_sim = 500000

AB_sim = model_AB.evidence().rvs(N_sim)
A_sim = AB_sim[:, :model_A.d]
B_sim = AB_sim[:, model_A.d:]

learning_rate = [1e-4]
architecture = [[5]*5, [5]*10, [10]*5, [10]*10, [25]*5]#, 
                #[25]*10, [50]*5]#, [50]*10, [100]*10]
arch_str = ['5x5', '5x10', '10x5', '10x10', '25x5']#,
            #'25x10', '50x5', '50x10', '100x10']
connections = []
for i in range(len(architecture)):
    arch = [model_A.d + model_B.d, *architecture[i], 1]
    connections.append(np.prod(arch))
argsort = np.argsort(connections)
architecture = [architecture[i] for i in argsort]
connections = [connections[i] for i in argsort]
activation = ['sigmoid', 'swish', 'tanh']#, 'relu', 'softplus']

import itertools
iters = list(itertools.product(learning_rate, 
                               architecture, activation))
iters = list(map(list, iters))

nsamp = 50
try:
    testing_data = np.loadtxt(base + 'testing_data.txt')
except FileNotFoundError:
    testing_data = None
data_train = None
accuracy = []
for i, (lr, arch, act) in enumerate(iters):

    nrei = nre(lr=lr)
    nrei.build_model(model_A.d+model_B.d,
                        arch, act)
    #nrei.build_compress_model(len(A_obs), len(B_obs),
    #                    [(len(A_obs) + len(B_obs))//2,
    #                     20, 20, 10], [25]*5, 'relu')
    if data_train is None:
        norm_data_train, norm_data_test, data_train, data_test, labels_train, labels_test = \
            simulation_process(A_sim, B_sim)
    else:
        pass
    nrei.data_test = norm_data_test
    nrei.labels_test = labels_test
    nrei.data_train = norm_data_train
    nrei.labels_train = labels_train
    nrei.simulation_func_A = None
    nrei.simulation_func_B = None

    model, data_test, labels_test = \
        nrei.training(epochs=30, patience=20,
                        batch_size=1000)
    
    nrei.save(base + f'lr_{lr}_arch_{arch}_act_{act}.pkl')

    if testing_data is None:
        testing_data = model_AB.evidence().rvs(nsamp)
        np.savetxt(base + 'testing_data.txt', testing_data)
    else:
        np.loadtxt(base + 'testing_data.txt')
    
    A_sim = testing_data[:, :model_A.d]
    B_sim = testing_data[:, model_A.d:]
    logr = logR(A_sim, B_sim)

    A_sim = (A_sim - data_train[:, :model_A.d].mean(axis=0)) / \
        data_train[:, :model_A.d].std(axis=0)
    B_sim = (B_sim - data_train[:, model_A.d:].mean(axis=0)) / \
        data_train[:, model_A.d:].std(axis=0)

    data_test = np.hstack([A_sim, B_sim])

    nrei.__call__(iters=testing_data)
    r = nrei.r_values
    mask = np.isfinite(r)
    r = r[mask]
    testing_data = testing_data[mask]

    k = ks_2samp(r, logr)
    

    accuracy.append(k.statistic)
    print('Architecture: ', arch, 'Activation: ', act, 'LR: ', lr)
    print('Accuracy: ', accuracy[-1])
accuracy = np.array(accuracy)


if len(learning_rate) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3), sharey=True)
    ax = axes.flatten()
    iters_lre3 = iters[:len(iters)//2]
    iters_lre4 = iters[len(iters)//2:]
    ae3 = np.reshape(accuracy[:len(accuracy)//2], (len(architecture), 
                                                        len(activation)))
    ae4 = np.reshape(accuracy[len(accuracy)//2:], (len(architecture),
                                                            len(activation)))

    cb = ax[0].imshow(ae3, cmap='viridis_r', aspect='auto', 
                    vmin=0, vmax=np.max([np.max(ae3), np.max(ae4)]))
    plt.colorbar(cb, ax=ax[0], 
        label=r'$\frac{2 |\log R^{95} - \log R_{obs}^{95}|}{\log R^{95} + \log R_{obs}^{95}}$')
    ax[0].set_xticks(range(len(activation)), activation, rotation=45)
    ax[0].set_yticks(range(len(connections)), 
                    ['{:.2f}'.format(np.log10(i)) for i in connections])
    ax[0].set_xlabel('Activation Function')
    ax[0].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
    ax[0].set_title('Learning Rate = 0.001')

    cb = ax[1].imshow(ae4, cmap='viridis_r', aspect='auto',
                    vmin=0, vmax=np.max([np.max(ae3), np.max(ae4)]))
    plt.colorbar(cb, ax=ax[1],
        label=r'$\frac{2 |\log R^{95} - \log R_{obs}^{95}|}{\log R^{95} + \log R_{obs}^{95}}$')
    ax[1].set_xticks(range(len(activation)), activation, rotation=45)
    ax[1].set_yticks(range(len(connections)), 
                    ['{:.2f}'.format(np.log10(i)) for i in connections])
    ax[1].set_xlabel('Activation Function')
    #ax[1].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
    ax[1].set_title('Learning Rate = 0.0001')
    plt.tight_layout()
    plt.savefig('parameter-sweep-accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
elif len(learning_rate) == 1:
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 4))
    ae4 = np.reshape(accuracy, (len(architecture), len(activation)))

    cb = ax[0].imshow(ae4, cmap='viridis_r', aspect='auto', 
                    vmin=0, vmax=np.max([np.max(ae4), np.max(ae4)]))
    plt.colorbar(cb, ax=ax[0],
        label=r'$\frac{2 |\log R^{95} - \log R_{obs}^{95}|}{\log R^{95} + \log R_{obs}^{95}}$')
    ax[0].set_xticks(range(len(activation)), activation, rotation=45)
    print(np.argsort(connections))
    ax[0].set_yticks(range(len(connections)), 
                  [arch_str[a] for a in np.argsort(connections)])
    ax[0].set_xlabel('Activation Function')
    ax[0].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
    ax[0].set_title('Learning Rate = 0.0001')
    

    accuracy = np.reshape(accuracy, (len(architecture), len(activation)))
    for i in range(len(activation)):
        ax[1].plot(np.arange(len(connections)), accuracy[:, i], marker='o', ls='-', 
                label=activation[i])
    ax[1].set_xticks(range(len(connections)), 
                  [arch_str[a] for a in np.argsort(connections)])
    ax[1].set_xlabel('Architecture')
    ax[1].set_ylabel(r'$\frac{|\log R^{95} - \log R_{obs}^{95}|}{\log R_{obs}^{95}}$')
    ax[1].set_title('Learning Rate = 0.0001')
    plt.legend()
    plt.tight_layout()
    plt.savefig('parameter-sweep-accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()



