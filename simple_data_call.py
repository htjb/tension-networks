import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    #nrei.build_compress_model(len(exp2_freq), len(exp1_freq), 1, 
    #                       [len(exp2_freq), len(exp2_freq), len(exp2_freq)//2, 50, 10], 
    #                       [len(exp1_freq), len(exp1_freq), len(exp1_freq)//2, 50, 10], 
    #                       [10, 10, 10, 10, 10],
    #                       'sigmoid')
    nrei.build_simulations(exp2, exp1, exp_prior, exp_prior, signal_prior, n=100000)
    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=2000)
    nrei.save('test_model.pkl')

plt.plot(nrei.loss_history)
plt.plot(nrei.test_loss_history)
plt.yscale('log')
plt.show()

nrei.__call__(iters=2000)
r = nrei.r_values
mask = np.isfinite(r)
sigr = tf.keras.layers.Activation('sigmoid')(r[mask])
c = 0
for i in range(len(sigr)):
    if sigr[i] < 0.75:
        c += 1

temperatures = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
Rs =[ -63.01687826,  -12.40474379,    9.77660272,  -17.25641865,  -33.974196,
  -87.30443093, -114.3616329,  -134.16389689, -157.02886038,] 
sigma_Rs = [0.21169332, 0.20684563, 0.21888224, 0.21734393, 0.21426394, 0.21660869,
 0.2113068,  0.22946067, 0.22718888]


plt.hist(r[mask], bins=25, label=f'{c/len(sigr)*100:.2f} % Mis-classified', color='C1')
plt.yticks([])
for i,t in enumerate(temperatures):
    plt.axvline(Rs[i], ls='--', label= f'{round(t/0.2, 2)}', color=plt.get_cmap('jet')(i/len(temperatures)))
    plt.axvspan(Rs[i] - sigma_Rs[i], Rs[i] + sigma_Rs[i], alpha=0.1, color=plt.get_cmap('jet')(i/len(temperatures)))
plt.xlabel(r'$\log R$')
plt.ylabel('Frequency')
plt.tight_layout()
plt.legend()
plt.savefig('test_r_hist.png', dpi=300)
plt.show()

from anesthetic import MCMCSamples
samples = MCMCSamples(data=r[mask], columns=['R'])
axes = samples.plot_1d('R', fc='C1', ec='k')
for i,t in enumerate(temperatures):
    plt.axvline(Rs[i], ls='--', label= f'{round(t/0.2, 2)}', color=plt.get_cmap('jet')(i/len(temperatures)))
    plt.axvspan(Rs[i] - sigma_Rs[i], Rs[i] + sigma_Rs[i], alpha=0.1, color=plt.get_cmap('jet')(i/len(temperatures)))
plt.xlim([-5, 25])
plt.xlabel(r'$\log R$')
plt.ylabel('Frequency')
plt.tight_layout()
plt.legend()
plt.savefig('test_r_kde.png', dpi=300)
plt.show()

from scipy.stats import ecdf

r  = np.sort(r[mask])
c = ecdf(r)

plt.plot(r, c.cdf.evaluate(r))
for i,t in enumerate(temperatures):
    print(t, c.cdf.evaluate(Rs[i]))
    plt.axhline(c.cdf.evaluate(Rs[i]), ls='--',
                label= f'{round(t/0.2, 2)}', 
                color=plt.get_cmap('jet')(i/len(temperatures)))
    plt.axhspan(c.cdf.evaluate(Rs[i] - sigma_Rs[i]), 
                c.cdf.evaluate(Rs[i] + sigma_Rs[i]), 
                alpha=0.1, 
                color=plt.get_cmap('jet')(i/len(temperatures)))
"""for i,t in enumerate(temperatures):
    plt.axvline(Rs[i], ls='--', 
                label= f'{round(t/0.2, 2)}', 
                color=plt.get_cmap('jet')(i/len(temperatures)))
    plt.axvspan(Rs[i] - sigma_Rs[i], Rs[i] + sigma_Rs[i],
                 alpha=0.1, 
                 color=plt.get_cmap('jet')(i/len(temperatures)))"""
plt.xlabel(r'$\log R$')
plt.ylabel(r'$P(\log R < \log R_{obs})$')
plt.legend()
plt.tight_layout()
plt.savefig('test_r_cdf.png', dpi=300)
plt.show()
sys.exit(1)

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
    elif p[i] > 0.25 and r[i] < 0.75 and labels_test[i] == 1:
        confused1 += 1
    elif p[i] > 0.25 and r[i] < 0.75 and labels_test[i] == 0:
        confused0 += 1

cm = [[correct0, wrong0, confused0],
        [correct1, wrong1, confused1]]

plt.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(3):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='k',
                 bbox=dict(facecolor='white', lw=0))
plt.xticks([0, 1, 2], ['Correct', 'Wrong', 'Confused'])
plt.yticks([0, 1], ['In tension', 'Not In Tension'])
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=300)
plt.show()

