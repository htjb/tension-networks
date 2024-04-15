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
    parameters[:, 0] = np.random.uniform(0.0, 4.0, n) #amp
    parameters[:, 1] = np.random.uniform(60.0, 90.0, n) #nu_0
    parameters[:, 2] = np.random.uniform(5.0, 40.0, n) #w
    parameters[:, 3] = np.random.uniform(0.001, 0.1, n) #sigma
    return parameters

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)

exp1 = signal_func_gen(exp1_freq)
exp2 = signal_func_gen(exp2_freq)

base = 'parameter-sweep/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME_POLY = True

lr = [1e-3, 1e-4]
architecture = [[5]*5, [5]*10, [10]*5]#, [10]*25, [25]*5, 
                #[25]*10, [50]*5, [50]*10, [100]*10]
connections = []
for i in range(len(architecture)):
    arch = [len(exp2_freq) + len(exp1_freq), *architecture[i], 1]
    connections.append(np.prod(arch))
argsort = np.argsort(connections)
architecture = [architecture[i] for i in argsort]
connections = [connections[i] for i in argsort]
activation = ['sigmoid']#, 'tanh', 'relu', 'softplus']

import itertools
iters = list(itertools.product(lr, architecture, activation))
iters = list(map(list, iters))

nsamp = 1000
testing_data = None
accuracy = []
for i, (lr, arch, act) in enumerate(iters):
    try:
        nrei = nre.load(base + f'lr_{lr}_arch_{arch}_act_{act}.pkl',
                        exp2, exp1, signal_prior)
    except FileNotFoundError:
        nrei = nre(lr=lr)
        nrei.build_model(len(exp2_freq) + len(exp1_freq),
                            arch, act)
        nrei.build_simulations(exp2, exp1, signal_prior, n=100000)
        model, data_test, labels_test = nrei.training(epochs=1000, 
                                                        batch_size=1000)
        nrei.save(base + f'lr_{lr}_arch_{arch}_act_{act}.pkl')

    if testing_data is None:
        testing_data = nrei.__call__(iters=nsamp)
    else:
        nrei.__call__(iters=testing_data)
    r = nrei.r_values
    mask = np.isfinite(r)
    r = r[mask]
    testing_data = testing_data[mask]
    print('Surviving Simulations: ', len(r)/nsamp*100, '%')

    #q95_arg = np.argsort(r)[len(r)//2]
    q95_arg = np.argmin(np.abs(r - np.quantile(r, 0.95)))

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

    accuracy.append(np.abs(r[q95_arg] - Rs.mean())/(r[q95_arg] + Rs.mean())*2)
    print('Accuracy: ', accuracy[-1], 'R: ', r[q95_arg], 'Rs: ', Rs.mean())
accuracy = np.array(accuracy)


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
plt.colorbar(cb, ax=ax[0], label=r'$\frac{|R - R_{obs}|}{R_{obs}}$')
ax[0].set_xticks(range(len(activation)), activation, rotation=45)
ax[0].set_yticks(range(len(connections)), 
                 ['{:.2f}'.format(np.log10(i)) for i in connections])
ax[0].set_xlabel('Activation Function')
ax[0].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
ax[0].set_title('Learning Rate = 0.001')

cb = ax[1].imshow(ae4, cmap='viridis_r', aspect='auto',
                  vmin=0, vmax=np.max([np.max(ae3), np.max(ae4)]))
plt.colorbar(cb, ax=ax[1], label=r'$\frac{|R - R_{obs}|}{R_{obs}}$')
ax[1].set_xticks(range(len(activation)), activation, rotation=45)
ax[1].set_yticks(range(len(connections)), 
                 ['{:.2f}'.format(np.log10(i)) for i in connections])
ax[1].set_xlabel('Activation Function')
#ax[1].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
ax[1].set_title('Learning Rate = 0.0001')
plt.tight_layout()
plt.savefig('parameter-sweep-accuracy.png', dpi=300, bbox_inches='tight')
plt.show()



