import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')


def signal_func_gen(freqs):
    def signal(_, parameters):
        amp, nu_0, w = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2)) + \
            np.random.normal(0, 0.025, len(freqs))
    return signal

def signal_prior(n):
    parameters = np.ones((n, 3))
    parameters[:, 0] = np.random.uniform(0.0, 4.0, n) #amp
    parameters[:, 1] = np.random.uniform(60.0, 90.0, n) #nu_0
    parameters[:, 2] = np.random.uniform(5.0, 40.0, n) #w
    return parameters

def exp_prior(n):
    """
    The way tensionnet is set up it requires some
    parameters that are unique to each experiment. Here I give an array of
    zeros because the experimetns are just signal plus noise. Doesn't have
    any impact on the results.
    """
    return np.zeros((n, 2))

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)

exp2 = signal_func_gen(exp2_freq)
exp1 = signal_func_gen(exp1_freq)

from tensionnet.tensionnet import nre

try:
    nrei = nre.load('test_model.pkl',
                exp2, exp1, exp_prior,
                exp_prior, signal_prior)
except:
    nrei = nre(lr=1e-4)
    nrei.build_model(len(exp2_freq) + len(exp1_freq), 1, 
                        [100]*10, 'sigmoid')
    nrei.build_simulations(exp2, exp1, exp_prior, exp_prior, signal_prior, n=100000)
    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=2000)
    nrei.save('test_model.pkl')

nrei.__call__(iters=5000)
r = nrei.r_values
mask = np.isfinite(r)
sigr = tf.keras.layers.Activation('sigmoid')(r[mask])
c = 0
good_idx = []
for i in range(len(sigr)):
    if sigr[i] < 0.75:
        c += 1
    else:
        good_idx.append(i)

r = r[good_idx]
mask = np.isfinite(r)

"""temperatures = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])/0.2
Rs =[ -63.01687826,  -12.40474379,    9.77660272,  -17.25641865,  -33.974196,
  -87.30443093, -114.3616329,  -134.16389689, -157.02886038,] 
sigma_Rs = [0.21169332, 0.20684563, 0.21888224, 0.21734393, 0.21426394, 0.21660869,
 0.2113068,  0.22946067, 0.22718888]"""

temperatures = np.array([0.15, 0.2, 0.25])/0.2
Rs =[-12.40474379,    9.77660272,  -17.25641865,] 
sigma_Rs = [0.20684563, 0.21888224, 0.21734393]

# get every other temperature
"""temperatures = temperatures[::2]
Rs = Rs[::2]
sigma_Rs = sigma_Rs[::2]"""

y_pos = [100]*len(temperatures)

fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.3))

for i, t in enumerate(temperatures):
        exp2_data_no_tension = np.loadtxt(f'toy_chains_temp_sweep/exp2_data_{t*0.2}.txt')
        axes[0, 0].plot(exp2_freq, exp2_data_no_tension, label=f'Exp. 2: {t*0.2} K', c='r', alpha=1/(i+1))

exp1_data = np.loadtxt('toy_chains_temp_sweep/exp1_data_no_tension.txt')
axes[0, 0].plot(exp1_freq, exp1_data, label='Exp. 1: 0.2 K', c='k')
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_xlabel('Frequency [MHz]')
axes[0, 0].set_ylabel(r'$\delta T_b$ [K]')


axes[1, 0].hist(r[mask], bins=25, color='C1')
axes[1, 0].set_yticks([])
for i,t in enumerate(temperatures):
        axes[1, 0].axvline(Rs[i], ls='--', c='r')
        axes[1, 0].axvspan(Rs[i] - sigma_Rs[i], Rs[i] + sigma_Rs[i], alpha=0.1, color='r')
        axes[1, 0].annotate(r'$A_2 = $' + f'{round(t, 2)}'+ r'$A_1$', (Rs[i], y_pos[i]), ha='center', va='center',
                    rotation=90, bbox=dict(color='w', ec='k'), fontsize=8)
axes[1, 0].set_xlabel(r'$\log R$')
axes[1, 0].set_ylabel('Frequency')

from scipy.stats import ecdf

r  = np.sort(r[mask])
c = ecdf(r)

axes[1, 1].plot(r, c.cdf.evaluate(r))
for i,t in enumerate(temperatures):
        if temperatures[i] == 1:
            axes[1, 1].axhline(c.cdf.evaluate(Rs[i]), ls='--',
                    color='r')
            axes[1, 1].axhspan(c.cdf.evaluate(Rs[i] - sigma_Rs[i]), 
                    c.cdf.evaluate(Rs[i] + sigma_Rs[i]), 
                    alpha=0.1, 
                    color='r')
            axes[1, 1].annotate(r'$A_2 = 1.0 A_1$', (Rs[i], c.cdf.evaluate(Rs[i])), ha='center', va='center',
                    bbox=dict(color='w', ec='k'), fontsize=8)
    
axes[1, 1].axhline(c.cdf.evaluate(Rs[0]), ls='--',
        color='r')
axes[1, 1].axhspan(c.cdf.evaluate(Rs[0] - sigma_Rs[i]), 
        c.cdf.evaluate(Rs[i] + sigma_Rs[i]), 
        alpha=0.1, 
        color='r')
axes[1, 1].annotate(r'In Tension. $A_2 \neq A_1$', (Rs[1], 0.), ha='center', va='center',
            bbox=dict(color='w', ec='k'), fontsize=8)
axes[1, 1].set_xlabel(r'$\log R$')
axes[1, 1].set_ylabel(r'$P(\log R < \log R_{obs})$')

idx = [int(np.random.uniform(0, len(nrei.labels_test), 1)) for i in range(1000)]

labels_test = nrei.labels_test[idx]

nrei.__call__(iters=nrei.data_test[idx])
p = tf.keras.layers.Activation('sigmoid')(nrei.r_values)
"""plt.hist(p, bins=25)
plt.show()"""

correct1, correct0, wrong1, wrong0, confused1, confused0 = 0, 0, 0, 0, 0, 0
for i in range(len(p)):
    if p[i] > 0.75 and labels_test[i] == 1:
        correct1 += 1
    elif p[i] < 0.25 and labels_test[i] == 0:
        correct0 += 1
    elif p[i] > 0.75 and labels_test[i] == 0:
        wrong0 += 1
    elif p[i] < 0.25 and labels_test[i] == 1:
        wrong1 += 1
    elif p[i] > 0.25 and p[i] < 0.75 and labels_test[i] == 1:
        confused1 += 1
    elif p[i] > 0.25 and p[i] < 0.75 and labels_test[i] == 0:
        confused0 += 1

total_0 = len(labels_test[labels_test == 0])
total_1 = len(labels_test[labels_test == 1])

cm = [[correct0/total_0*100, wrong0/total_0*100, confused0/total_0*100],
        [correct1/total_1*100, wrong1/total_1*100, confused1/total_1*100]]


axes[0, 1].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(3):
        axes[0, 1].text(j, i, '{:.2f} \%'.format(cm[i][j]), ha='center', va='center', color='k',
                 bbox=dict(facecolor='white', lw=0))
axes[0, 1].set_xticks([0, 1, 2], ['Correct', 'Wrong', 'Confused'])
axes[0, 1].set_yticks([0, 1], ['In tension', 'Not In Tension'])
plt.tight_layout()
plt.savefig('toy_example.pdf', bbox_inches='tight', dpi=300)
plt.show()

