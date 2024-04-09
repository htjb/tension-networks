import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from matplotlib import rc
from tensionnet.tensionnet import nre
from scipy.stats import ecdf, norm
from tensionnet.utils import calcualte_stats

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

exp2 = signal_func_gen(exp2_freq)
exp1 = signal_func_gen(exp1_freq)

base = 'parameter-sweep/'

lr = [1e-3, 1e-4]
architecture = [[5]*5, [5]*10, [10]*5, [10]*25, [25]*5, 
                [25]*10, [50]*5, [50]*10, [100]*10]
connections = []
for i in range(len(architecture)):
    arch = [len(exp2_freq) + len(exp1_freq), *architecture[i], 1]
    connections.append(np.prod(arch))
activation = ['sigmoid', 'tanh', 'relu']

import itertools
iters = list(itertools.product(lr, architecture, activation))
iters = list(map(list, iters))
print(len(iters))
print(iters)

nsamp = 5000
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
                                                        batch_size=5000)
        nrei.save(base + f'lr_{lr}_arch_{arch}_act_{act}.pkl')

    if testing_data is None:
        testing_data = nrei.__call__(iters=nsamp)
    else:
        nrei.__call__(iters=testing_data)
    r = nrei.r_values
    mask = np.isfinite(r)
    sigr = tf.keras.layers.Activation('sigmoid')(r[mask])
    good_idx = sigr > 0.75
    r = r[good_idx]
    print('Surviving Simulations: ', len(r)/nsamp*100, '%')
    accuracy.append(len(r)/nsamp)
accuracy = np.array(accuracy)


fig, axes = plt.subplots(1, 2)
ax = axes.flatten()
iters_lre3 = iters[:len(iters)//2]
iters_lre4 = iters[len(iters)//2:]
ae3 = np.reshape(accuracy[:len(accuracy)//2], (len(architecture), 
                                                     len(activation)))
ae4 = np.reshape(accuracy[len(accuracy)//2:], (len(architecture),
                                                        len(activation)))

cb = ax[0].imshow(ae3, cmap='viridis')
plt.colorbar(cb, ax=ax[0])
ax[0].set_xticks(range(len(activation)), activation)
ax[0].set_yticks(range(len(connections)), 
                 ['{:.2f}'.format(np.log10(i)) for i in connections])
ax[0].set_xlabel('Activation Function')
ax[0].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
ax[0].set_title('Learning Rate: 1e-3')

cb = ax[1].imshow(ae4, cmap='viridis')
plt.colorbar(cb, ax=ax[1])
ax[1].set_xticks(range(len(activation)), activation)
ax[1].set_yticks(range(len(connections)), 
                 ['{:.2f}'.format(np.log10(i)) for i in connections])
ax[1].set_xlabel('Activation Function')
ax[1].set_ylabel(r'$\log_{10}($' + 'No. Weights' + r'$)$')
ax[1].set_title('Learning Rate: 1e-4')

plt.savefig('parameter-sweep-accuracy.png')
plt.show()



