import numpy as np
import matplotlib.pyplot as plt

def signal_func_gen(freqs):
    def signal(_, parameters):
        amp, nu_0, w = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2))
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

true_params = np.array([0.2, 78.0, 10.0])

# I don't need this here... but once this is working I can
# use this to generate the data for the experiments. Should generate 
# a in tension and not in tension experimental data sets and see where they fall
# in the distribution predicted by the network.
"""exp1_data = signal_func_gen(exp1_freq)([None], true_params) \
    + np.random.normal(0, 0.03, 100)
exp2_data = signal_func_gen(exp2_freq)([None], true_params) \
    + np.random.normal(0, 0.025, 100)"""

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
                     [100]*5, 'sigmoid')
    #nrei.default_nn_model(len(exp23_freq) + len(exp1_freq))
    #nrei.build_compress_model(len(exp2_freq), len(exp1_freq), 1, 
    #                       [len(exp2_freq), len(exp2_freq), len(exp2_freq)//2, 50, 10], 
    #                       [len(exp1_freq), len(exp1_freq), len(exp1_freq)//2, 50, 10], 
    #                       [10, 10, 10, 10, 10],
    #                       'sigmoid')
    nrei.build_simulations(exp2, exp1, exp_prior, exp_prior, signal_prior, n=500000)
    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=1000)
    nrei.save('test_model.pkl')

plt.plot(nrei.loss_history)
plt.plot(nrei.test_loss_history)
plt.show()

nrei.__call__(iters=5000)
r = nrei.r_values
mask = np.isfinite(r)
plt.hist(np.log10(r[mask]), bins=20)
plt.axvline(7.89, color='k', ls='--', label='Out of tension example')
plt.axvline(-34.26, color='k', ls=':', label='In tension example')
plt.xlabel(r'$\log R$')
plt.ylabel('Frequency')
plt.tight_layout()
plt.legend()
plt.savefig('test_r_hist.png', dpi=300)
plt.show()

idx = [int(np.random.uniform(0, len(r), 1)) for i in range(1000)]

labels_test = nrei.labels_test[idx]

nrei.__call__(iters=nrei.data_test[idx])
r = nrei.raw_r

correct1, correct0, wrong1, wrong0, confused1, confused0 = 0, 0, 0, 0, 0, 0
for i in range(len(r)):
    if r[i] > 0.75 and labels_test[i] == 1:
        correct1 += 1
    elif r[i] < 0.25 and labels_test[i] == 0:
        correct0 += 1
    elif r[i] > 0.75 and labels_test[i] == 0:
        wrong0 += 1
    elif r[i] < 0.25 and labels_test[i] == 1:
        wrong1 += 1
    elif r[i] > 0.25 and r[i] < 0.75 and labels_test[i] == 1:
        confused1 += 1
    elif r[i] > 0.25 and r[i] < 0.75 and labels_test[i] == 0:
        confused0 += 1

cm = [[correct0, wrong0, confused0],
        [wrong1, correct1, confused1]]

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

