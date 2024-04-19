import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from matplotlib import rc
from tensionnet.tensionnet import nre
from scipy.stats import ecdf, norm
from tensionnet.utils import calcualte_stats, twentyone_example
from tensionnet.robs import run_poly
from anesthetic import read_chains
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


def signal_func_gen(freqs):
    def signal(parameters):
        amp, nu_0, w, sigma = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2)) + \
            np.random.normal(0, sigma, len(freqs))
    return signal

def signal_prior(n):
    parameters = np.ones((n, 4))
    parameters[:, 0] = np.random.uniform(0.0, 1.0, n) #amp
    parameters[:, 1] = np.random.uniform(70.0, 85.0, n) #nu_0
    parameters[:, 2] = np.random.uniform(5.0, 15.0, n) #w
    parameters[:, 3] = np.random.uniform(0.01, 0.1, n) #sigma
    return parameters

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)

exp1 = signal_func_gen(exp1_freq)
exp2 = signal_func_gen(exp2_freq)

base = 'parameter-sweep_nsims500000/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME_POLY = True

learning_rate = [1e-4]
architecture = [[5]*5, [5]*10, [10]*5, [10]*10, [25]*5]#, 
                #[25]*10, [50]*5]#, [50]*10, [100]*10]
arch_str = ['5x5', '5x10', '10x5', '10x10', '25x5']#,
            #'25x10', '50x5', '50x10', '100x10']
connections = []
for i in range(len(architecture)):
    arch = [len(exp2_freq) + len(exp1_freq), *architecture[i], 1]
    connections.append(np.prod(arch))
argsort = np.argsort(connections)
architecture = [architecture[i] for i in argsort]
connections = [connections[i] for i in argsort]
activation = ['sigmoid', 'swish', 'tanh']#, 'relu', 'softplus']

import itertools
iters = list(itertools.product(learning_rate, 
                               architecture, activation))
iters = list(map(list, iters))

nsamp = 5000
try:
    testing_data = np.loadtxt(base + 'testing_data.txt')
except FileNotFoundError:
    testing_data = None
data_train = None
accuracy = []
for i, (lr, arch, act) in enumerate(iters):
    try:
        nrei = nre.load(base + f'lr_{lr}_arch_{arch}_act_{act}.pkl',
                        exp2, exp1, signal_prior)
        if data_train is None:
            data_train = nrei.data_train
            data_test = nrei.data_test
            labels_train = nrei.labels_train
            labels_test = nrei.labels_test
    except FileNotFoundError:
        nrei = nre(lr=lr)
        nrei.build_model(len(exp2_freq) + len(exp1_freq),
                            arch, act)
        if data_train is None:
            nrei.build_simulations(exp2, exp1, signal_prior, n=500000)
            data_train = nrei.data_train
            data_test = nrei.data_test
            labels_train = nrei.labels_train
            labels_test = nrei.labels_test
        else:
            nrei.data_train = data_train
            nrei.data_test = data_test
            nrei.labels_train = labels_train
            nrei.labels_test = labels_test
            nrei.simulation_func_A = exp2
            nrei.simulation_func_B = exp1
            nrei.shared_prior = signal_prior
            nrei.prior_function_A = None
            nrei.prior_function_B = None
        model, data_test, labels_test = nrei.training(epochs=1000, 
                                                        batch_size=1000)
        nrei.save(base + f'lr_{lr}_arch_{arch}_act_{act}.pkl')

    if testing_data is None:
        testing_data = nrei.__call__(iters=nsamp)
        np.savetxt(base + 'testing_data.txt', testing_data)
    else:
        nrei.__call__(iters=testing_data)
    r = nrei.r_values
    mask = np.isfinite(r)
    r = r[mask]
    testing_data = testing_data[mask]
    print('Surviving Simulations: ', len(r)/nsamp*100, '%')

    
    #q95_arg = np.argmin(np.abs(r - np.quantile(r, 0.95)))
    q95_arg = np.argmin(np.abs(r - np.median(r)))

    q95_data = testing_data[q95_arg]
    q95_exp2 = q95_data[:len(exp2_freq)]*nrei.data_train.std(axis=0)[:len(exp2_freq)] \
        + nrei.data_train.mean(axis=0)[:len(exp2_freq)]
    q95_exp1 = q95_data[len(exp2_freq):]*nrei.data_train.std(axis=0)[len(exp2_freq):] \
        + nrei.data_train.mean(axis=0)[len(exp2_freq):]

    """plt.plot(exp2_freq, q95_exp2, label='Exp2')
    plt.plot(exp1_freq, q95_exp1, label='Exp1')

    plt.legend()
    plt.show()
    sys.exit(1)"""

    signal_poly_prior, \
        joint_prior, exp1likelihood, exp2likelihood, jointlikelihood = \
        twentyone_example(q95_exp1, q95_exp2, exp1_freq, exp2_freq)
    
    run_poly(signal_poly_prior, exp1likelihood, base + 
             f'exp1_q95_lr_{lr}_arch_{arch}_act_{act}', 
         nlive=100, RESUME=RESUME_POLY, nDims=4)
    
    run_poly(joint_prior, jointlikelihood, 
             base + f'joint_q95_lr_{lr}_arch_{arch}_act_{act}', 
             nlive=125, RESUME=RESUME_POLY, nDims=5)
    run_poly(signal_poly_prior, exp2likelihood, 
             base + f'exp2_q95_lr_{lr}_arch_{arch}_act_{act}', 
             nlive=100, RESUME=RESUME_POLY, nDims=4)

    exp1_samples = read_chains(base + 
                    f'exp1_q95_lr_{lr}_arch_{arch}_act_{act}/test')
    exp2_samples = read_chains(base + 
                    f'exp2_q95_lr_{lr}_arch_{arch}_act_{act}/test')
    joint_samples = read_chains(base + 
                    f'joint_q95_lr_{lr}_arch_{arch}_act_{act}/test')
    
    Rs = joint_samples.logZ(1000) - \
        exp1_samples.logZ(1000) - exp2_samples.logZ(1000)
    RS95 = np.quantile(Rs.values, 0.95)

    accuracy.append(np.abs(r[q95_arg] - RS95)/RS95)
    print('Architecture: ', arch, 'Activation: ', act, 'LR: ', lr)
    print('Accuracy: ', accuracy[-1], 'R: ', r[q95_arg], 'Rs: ', RS95)
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



