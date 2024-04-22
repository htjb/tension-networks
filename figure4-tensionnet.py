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

base = 'chains/21cm_temp_sweep/'

try:
    nrei = nre.load('figure4-nre.pkl',
                exp2, exp1, signal_prior)
except:
    nrei = nre(lr=1e-4)
    nrei.build_model(len(exp2_freq) + len(exp1_freq),
                        [25]*5, 'sigmoid')
    nrei.build_simulations(exp2, exp1, signal_prior, n=200000)
    model, data_test, labels_test = nrei.training(epochs=1000, 
                                                  batch_size=1000)
    nrei.save('figure4-nre.pkl')

nrei.__call__(iters=5000)
r = nrei.r_values
mask = np.isfinite(r)
"""sig = tf.keras.activations.sigmoid(r[mask])
mask = sig > 0.75"""
r = r[mask]

temperatures = np.array([0.15, 0.2, 0.25])/0.2

Rss = np.loadtxt(base + 'Rs.txt')
Rs, sigma_Rs = Rss[:, 0], Rss[:, 1]

y_pos = [200]*len(temperatures)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

for i, t in enumerate(temperatures):
        exp2_data_no_tension = np.loadtxt(base + 
             f'exp2_data_{t*0.2}.txt')
        axes[0].plot(exp2_freq, exp2_data_no_tension, 
                        label=f'Exp. B: {t*0.2} K', c='r', alpha=1/(i+1))

exp1_data = np.loadtxt(base + 'exp1_data_truth.txt')
axes[0].plot(exp1_freq, exp1_data, label='Exp. A: 0.2 K', c='k')
axes[0].legend(fontsize=8)
axes[0].set_xlabel('Frequency [MHz]')
axes[0].set_ylabel(r'$\delta T_b$ [K]')

axes[1].hist(r, bins=25, color='C0')
axes[1].set_yticks([])
for i,t in enumerate(temperatures):
        axes[1].axvline(Rs[i], ls='--', c='r')
        axes[1].axvspan(Rs[i] - sigma_Rs[i], 
                           Rs[i] + sigma_Rs[i], alpha=0.1, color='r')
        axes[1].annotate(r'$A_B = $' + 
                            f'{round(t, 2)}'+ 
                            r'$A_A$', (Rs[i], y_pos[i]), 
                            ha='center', va='center',
                    rotation=90, bbox=dict(color='w', ec='k'), fontsize=8)
axes[1].set_xlabel(r'$\log R$')
axes[1].set_ylabel(r'$P(\log R)$')
r  = np.sort(r)
c = ecdf(r)

axes[2].plot(r, c.cdf.evaluate(r))
for i,t in enumerate(temperatures):
        sigmaD, sigma_D_upper, sigma_D_lower, \
            sigmaA, sigma_A_upper, sigma_A_lower, \
                sigmaR, sigmaR_upper, sigmaR_lower = \
                    calcualte_stats(Rs[i], sigma_Rs[i], c)
        print(f'Temp: {temperatures[i]}')
        print(f'Rs: {Rs[i]}, Rs_upper: {Rs[i] + sigma_Rs[i]},' + 
              f'Rs_lower: {Rs[i] - sigma_Rs[i]}')
        print(f'sigmaD: {sigmaD}, sigma_D_upper: ' + 
              f'{np.abs(sigmaD - sigma_D_upper)}, ' +
              f'sigma_D_lower: {np.abs(sigma_D_lower - sigmaD)}')
        print(f'sigmaA: {sigmaA}, sigma_A_upper: ' +
                f'{np.abs(sigmaA - sigma_A_upper)}, ' +
                f'sigma_A_lower: {np.abs(sigma_A_lower - sigmaA)}')
        print(f'sigmaR: {sigmaR}, sigmaR_upper: ' + 
              f'{np.abs(sigmaR - sigmaR_upper)}, ' +
              f'sigmaR_lower: {np.abs(sigmaR_lower - sigmaR)}')
        np.savetxt(base + f'tension_stats_{t*0.2}.txt',
                   np.hstack([sigmaD, sigma_D_upper, sigma_D_lower,
                              sigmaA, sigma_A_upper, sigma_A_lower,
                              sigmaR, sigmaR_upper, sigmaR_lower]).T)
        if temperatures[i] == 1:
            axes[2].axhline(c.cdf.evaluate(Rs[i]), ls='--',
                    color='r')
            axes[2].axhspan(c.cdf.evaluate(Rs[i] - sigma_Rs[i]), 
                    c.cdf.evaluate(Rs[i] + sigma_Rs[i]), 
                    alpha=0.1, 
                    color='r')
            axes[2].annotate(r'$A_B = 1.0 A_A$', 
                    (Rs[1]/2, c.cdf.evaluate(Rs[i])), ha='center', va='center',
                    bbox=dict(color='w', ec='k'), fontsize=8)
    
axes[2].axhline(c.cdf.evaluate(Rs[0]), ls='--',
        color='r')
axes[2].axhspan(c.cdf.evaluate(Rs[0] - sigma_Rs[i]), 
        c.cdf.evaluate(Rs[i] + sigma_Rs[i]), 
        alpha=0.1, 
        color='r')
axes[2].annotate(r'In Tension. $A_B \neq A_A$', 
                    (Rs[1]/2, 0.), ha='center', va='center',
            bbox=dict(color='w', ec='k'), fontsize=8)
axes[2].set_xlabel(r'$\log R$')
axes[2].set_ylabel(r'$P(\log R < \log R^\prime)$')

plt.tight_layout()
plt.savefig('figures/figure4.pdf', bbox_inches='tight', dpi=300)
plt.show()

