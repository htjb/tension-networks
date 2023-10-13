import numpy as np
import matplotlib.pyplot as plt

"""def simulator(simulation_func_A, simulation_func_B,
                prior_function_A, prior_function_B,
                shared_prior, n=10000, call_type='train'):

    # generate lots of simulations 
    simsA, params = [], []
    simsB = []
    for i in range(n):
        simsA.append(simulation_func_A(thetaA[i], thetaShared[i]))
        simsB.append(simulation_func_B(thetaB[i], thetaShared[i]))
        params.append([*thetaA[i], *thetaB[i], *thetaShared[i]])
    simsA = np.array(simsA)
    simsB = np.array(simsB)
    params = np.array(params)

    #simsA = (simsA - simsA.mean(axis=0)) / simsA.std(axis=0)
    #simsB = (simsB - simsB.mean(axis=0)) / simsB.std(axis=0)

    simsA = (simsA - simsA.min(axis=0)) / (simsA.max(axis=0) - simsA.min(axis=0))
    simsB = (simsB - simsB.min(axis=0)) / (simsB.max(axis=0) - simsB.min(axis=0))

    idx = np.arange(0, n, 1)
    shuffle(idx)
    mis_labeled_simsB = simsB[idx]

    data = []
    for i in range(n):
        data.append([*simsA[i], *simsB[i], 1])
        if call_type == 'train':
            data.append([*simsA[i], *mis_labeled_simsB[i], 0])
    data = np.array(data)

    idx = np.arange(0, 2*n, 1)
    if call_type == 'train':
        shuffle(idx)
        input_data = data[idx, :-1]
        labels = data[idx, -1]
    elif call_type == 'eval':
        input_data = data[:, :-1]
        labels = data[:, -1]

    return input_data, labels"""

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
    """
    The way tensionnet is set up it requires some
    parameters that are unique to each experiment. Here I give an array of
    zeros because the experimetns are just signal plus noise. Doesn't have
    any impact on the results.
    """
    return np.zeros((n, 2))

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
true_params = np.array([0.2, 78.0, 10.0, 1.0])

exp1_data = signal_func_gen(exp1_freq)([None], true_params) + np.random.normal(0, 0.03, 100)
exp2_data = signal_func_gen(exp2_freq)([None], true_params) + np.random.normal(0, 0.025, 100)

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
model, data_test, labels_test = nrei.training(epochs=1000, batch_size=1000)

plt.plot(nrei.loss_history)
plt.plot(nrei.test_loss_history)
plt.show()

nrei.__call__()
print(nrei.r_values)
plt.hist(nrei.r_values, bins=50)
plt.show()