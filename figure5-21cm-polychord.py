import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from anesthetic import read_chains
from tensionnet.tensionnet import nre
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import os
import time

def signal_func_gen(freqs):
    def signal(parameters):
        amp, nu_0, w = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2))
    return signal

def signal_poly_prior(cube):
    theta = np.zeros(4)
    theta[0] = UniformPrior(0, 4)(cube[0]) #amp
    theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
    theta[2] = UniformPrior(5, 40)(cube[2]) #w
    theta[3] = UniformPrior(0.001, 0.1)(cube[3]) #sigma
    return theta

def joint_prior(cube):
    theta = np.zeros(5)
    theta[0] = UniformPrior(0, 4)(cube[0]) #amp
    theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
    theta[2] = UniformPrior(5, 40)(cube[2]) #w
    theta[3] = UniformPrior(0.001, 0.1)(cube[3]) #sigma1
    theta[4] = UniformPrior(0.001, 0.1)(cube[4]) #sigma2
    return theta

def exp1likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[-1]**2) \
        - 0.5 * (exp1_data - exp1_sf(theta[:-1]))**2/theta[-1]**2).sum(),[]

def exp2likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[-1]**2) \
        - 0.5 * (exp2_data - exp2_sf(theta[:-1]))**2/theta[-1]**2).sum(),[]

def jointlikelihood(theta):
    return exp1likelihood(theta[:-1])[0] + \
        exp2likelihood([*theta[:-2], theta[-1]])[0], []

base = 'chains/21cm_temp_sweep/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME = False

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)

temperatures = [0.15, 0.2, 0.25]
true_params = np.array([0.2, 78.0, 10.0])

try:
    exp1_data = np.loadtxt(base + 'exp1_data_truth.txt')
except:
    exp1_data = exp1_sf(true_params) \
        + np.random.normal(0, 0.025, 100)
    np.savetxt(base + 'exp1_data_truth.txt', exp1_data)

"""exp2_data = np.loadtxt(base + f'exp2_data_{0.2}.txt')
timesl1, timesl2, timesJ = [], [], []
for i in range(100):
    s = time.time()
    print(exp1likelihood(signal_poly_prior(np.random.uniform(0, 1, 4))))
    e = time.time()
    timesl1.append(e-s)
    
    s = time.time()
    print(exp2likelihood(signal_poly_prior(np.random.uniform(0, 1, 4))))
    e = time.time()
    timesl2.append(e-s)
    
    s = time.time()
    print(jointlikelihood(joint_prior(np.random.uniform(0, 1, 5))))
    e = time.time()
    timesJ.append(e-s)

print(f'Average time for exp1likelihood: {np.mean(timesl1)}')
print(f'Average time for exp2likelihood: {np.mean(timesl2)}')
print(f'Average time for jointlikelihood: {np.mean(timesJ)}')
exit()"""

run_poly(signal_poly_prior, exp1likelihood, base + f'exp1', 
         nlive=100, RESUME=RESUME, nDims=4)
exp1_samples = read_chains(base + f'exp1/test')

Rs = []
for t in temperatures:
    try:
        exp2_data = np.loadtxt(base + f'exp2_data_{t}.txt')
    except:
        exp2_data = exp2_sf([t, 78.0, 10.0]) \
            + np.random.normal(0, 0.025, 100)
        np.savetxt(base + f'exp2_data_{t}.txt', exp2_data)

    run_poly(joint_prior, jointlikelihood, 
             base + f'joint_{t}', nlive=125, RESUME=RESUME, nDims=5)
    run_poly(signal_poly_prior, exp2likelihood, 
             base + f'exp2_{t}', nlive=100, RESUME=RESUME, nDims=4)

    exp2_samples = read_chains(base + f'exp2_{t}/test')
    joint_samples = read_chains(base + f'joint_{t}/test')

    Rs.append(joint_samples.logZ(1000) - 
              exp1_samples.logZ(1000) - exp2_samples.logZ(1000))

print(temperatures)
np.savetxt(base + 'Rs.txt', 
           np.vstack([np.mean(Rs, axis=1), np.std(Rs, axis=1)]).T)
