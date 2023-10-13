import numpy as np
import matplotlib.pyplot as plt

def signal_func_gen(freqs):
    def signal(_, parameters):
        amp, nu_0, w, tau = parameters
        b = ((4 * (freqs - nu_0) ** 2 / w ** 2) *
             np.log(-np.log((1 + np.exp(-tau)) / 2) / tau))
        return -amp * ((1 - np.exp(- tau * np.exp(b))) / (1 - np.exp(-tau)))
    return signal

def signal_prior(n):
    parameters = np.ones((n, 4))
    parameters[:, 0] = np.random.uniform(0.0, 4.0, n) #amp
    parameters[:, 1] = np.random.uniform(60.0, 90.0, n) #nu_0
    parameters[:, 2] = np.random.uniform(5.0, 40.0, n) #w
    parameters[:, 3] = np.random.uniform(0.0, 40.0, n) #tau
    return parameters

def exp_prior(n):
    return [None]

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
true_params = np.array([0.2, 78.0, 10.0, 1.0])

exp1 = signal_func_gen(exp1_freq)(None, true_params) + np.random.normal(0, 0.03, 100)
exp2 = signal_func_gen(exp2_freq)(None, true_params) + np.random.normal(0, 0.025, 100)

exp2 = signal_func_gen(exp2_freq)
exp1 = signal_func_gen(exp1_freq)

from tensionnet.tensionnet import nre

nrei = nre(lr=1e-4)
nrei.build_model(len(exp2_freq) + len(exp1_freq), 1, [100]*5, 'relu')
#nrei.default_nn_model(len(exp23_freq) + len(exp1_freq))
#nrei.build_compress_model(len(exp23_freq), len(exp1_freq), 1, 
#                       [len(exp23_freq), len(exp23_freq), len(exp23_freq)//2, 50, 10], 
#                       [len(exp1_freq), len(exp1_freq), len(exp1_freq)//2, 50, 10], 
#                       [10, 10, 10, 10, 10],
#                       'relu')
nrei.build_simulations(exp2, exp1, exp_prior, exp_prior, signal_prior, n=500000)
sys.exit(1)
model, data_test, labels_test = nrei.training(epochs=1000, batch_size=1000)

plt.plot(nrei.loss_history)
plt.plot(nrei.test_loss_history)
plt.show()

nrei.__call__()
print(nrei.r_values)
plt.hist(nrei.r_values, bins=50)
plt.show()