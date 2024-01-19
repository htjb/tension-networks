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

def exp1likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[3]**2) \
        - 0.5 * (exp1_data - exp1_sf([None], theta[:3]))**2/theta[3]**2).sum(),[]

def exp2likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[3]**2) \
        - 0.5 * (exp2_data - exp2_sf([None], theta[:3]))**2/theta[3]**2).sum(),[]

def jointlikelihood(theta):
    exp1theta = theta[:4]
    exp2theta = np.concatenate((theta[:3], theta[4:]))
    return exp1likelihood(exp1theta)[0] + exp2likelihood(exp2theta)[0], []

def build_priors(prior_bounds):
    def signal_poly_prior(cube):
        theta = np.zeros(4)
        theta[0] = UniformPrior(**prior_bounds[0])(cube[0]) #amp
        theta[1] = UniformPrior(**prior_bounds[1])(cube[1]) #nu_0
        theta[2] = UniformPrior(**prior_bounds[2])(cube[2]) #w
        theta[3] = LogUniformPrior(0.001, 0.1)(cube[3]) #noise
        return theta

    def joint_prior(cube):
        theta = np.zeros(5)
        theta[0] = UniformPrior(**prior_bounds[0])(cube[0]) #amp
        theta[1] = UniformPrior(**prior_bounds[1])(cube[1]) #nu_0
        theta[2] = UniformPrior(**prior_bounds[2])(cube[2]) #w
        theta[3] = LogUniformPrior(0.001, 0.1)(cube[3]) #exp1 noise
        theta[4] = LogUniformPrior(0.001, 0.1)(cube[4]) #exp2 noise
        return theta
    return signal_poly_prior, joint_prior


base = 'toy_chains_priors/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME = True

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)

true_params = np.array([0.2, 78.0, 10.0])

wide_prior_bounds = np.array([[0.0, 4.0], [60.0, 90.0], [5.0, 40.0]])
conservative_prior_bounds = np.array([[0.5, 1.0], [70.0, 80.0], [5.0, 15.0]])
narrow_prior_bounds = np.array([[0.2, 0.3], [76.0, 80.0], [8.0, 12.0]])

prior_sets = [wide_prior_bounds, conservative_prior_bounds, narrow_prior_bounds]
prior_sets_names = ['wide', 'conservative', 'narrow']
Rs = []
for i, ps in enumerate(prior_sets):
    signal_prior, joint_prior = build_priors(ps)
    base += prior_sets_names[i] + '/'
    if not os.path.exists(base):
        os.mkdir(base)

    try:
        exp1_data = np.loadtxt(base + 'exp1_data.txt')
    except:
        exp1_data = exp1_sf([None], true_params) \
            + np.random.normal(0, 0.025, 100)
        np.savetxt(base + 'exp1_data.txt', exp1_data)

    run_poly(signal_prior, exp1likelihood, base + f'exp1', nlive=1000, RESUME=RESUME)
    exp1_samples = read_chains(base + f'exp1/test')

    try:
        exp2_data = np.loadtxt(base + f'exp2_data.txt')
    except:
        exp2_data = exp2_sf([None], true_params) \
            + np.random.normal(0, 0.025, 100)
        np.savetxt(base + f'exp2_data.txt', exp2_data)

    run_poly(joint_prior, jointlikelihood, base + f'joint', RESUME=RESUME, nDims=5)
    run_poly(signal_prior, exp2likelihood, base + f'exp2', RESUME=RESUME)

    exp2_samples = read_chains(base + f'exp2/test')
    joint_samples = read_chains(base + f'joint/test')

    Rs.append(joint_samples.logZ(1000) - 
              exp1_samples.logZ(1000) - exp2_samples.logZ(1000))


print(np.mean(Rs, axis=1), np.std(Rs, axis=1))
