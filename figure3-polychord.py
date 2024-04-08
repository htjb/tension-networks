import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from anesthetic import read_chains
from tensionnet.tensionnet import nre
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import os

def signal_func_gen(freqs):
    def signal(_, parameters):
        amp, nu_0, w = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2))
    return signal

def signal_poly_prior(cube):
    theta = np.zeros(3)
    theta[0] = UniformPrior(0, 4)(cube[0]) #amp
    theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
    theta[2] = UniformPrior(5, 40)(cube[2]) #w
    return theta

def joint_prior(cube):
    theta = np.zeros(3)
    theta[0] = UniformPrior(0, 4)(cube[0]) #amp
    theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
    theta[2] = UniformPrior(5, 40)(cube[2]) #w
    return theta

def exp1likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*0.025**2) \
        - 0.5 * (exp1_data - exp1_sf([None], theta))**2/0.025**2).sum(),[]

def exp2likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*0.025**2) \
        - 0.5 * (exp2_data - exp2_sf([None], theta))**2/0.025**2).sum(),[]

def jointlikelihood(theta):
    return exp1likelihood(theta)[0] + exp2likelihood(theta)[0], []

base = 'toy_chains_temp_sweep/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME = True

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)

temperatures = [0.15, 0.2, 0.25]
true_params = np.array([0.2, 78.0, 10.0])

try:
    exp1_data = np.loadtxt(base + 'exp1_data_no_tension.txt')
except:
    exp1_data = exp1_sf([None], true_params) \
        + np.random.normal(0, 0.025, 100)
    np.savetxt(base + 'exp1_data_no_tension.txt', exp1_data)

run_poly(signal_poly_prior, exp1likelihood, base + f'test_exp1', nlive=1000, RESUME=RESUME)
exp1_samples = read_chains(base + f'test_exp1/test')

Rs = []
for t in temperatures:
    try:
        exp2_data = np.loadtxt(base + f'exp2_data_{t}.txt')
    except:
        exp2_data = exp2_sf([None], [t, 78.0, 10.0]) \
            + np.random.normal(0, 0.025, 100)
        np.savetxt(base + f'exp2_data_{t}.txt', exp2_data)

    run_poly(joint_prior, jointlikelihood, base + f'test_joint_{t}', nlive=1000, RESUME=RESUME)
    run_poly(signal_poly_prior, exp2likelihood, base + f'test_exp2_{t}', nlive=1000, RESUME=RESUME)

    exp2_samples = read_chains(base + f'test_exp2_{t}/test')
    joint_samples = read_chains(base + f'test_joint_{t}/test')

    Rs.append(joint_samples.logZ(1000) - 
              exp1_samples.logZ(1000) - exp2_samples.logZ(1000))

print(temperatures)
print(np.mean(Rs, axis=1), np.std(Rs, axis=1))
