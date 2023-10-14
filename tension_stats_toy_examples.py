import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from anesthetic import read_chains
from tensionnet.tensionnet import nre
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior

def signal_func_gen(freqs):
    def signal(_, parameters):
        amp, nu_0, w = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2))
    return signal

def signal_poly_prior(cube):
    theta = np.zeros(4)
    theta[0] = UniformPrior(0, 4)(cube[0]) #amp
    theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
    theta[2] = UniformPrior(5, 40)(cube[2]) #w
    theta[3] = LogUniformPrior(0.001, 0.1)(cube[3]) #noise
    return theta

def exp1likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[3]**2) \
        - 0.5 * (exp1_data - exp1_sf([None], theta[:3]))**2/theta[3]**2).sum(),[]

def exp2likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[3]**2) \
        - 0.5 * (exp2_data - exp2_sf([None], theta[:3]))**2/theta[3]**2).sum(),[]

def jointlikelihood(theta):
    return exp1likelihood(theta)[0] + exp2likelihood(theta)[0], []


exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)

true_params = np.array([0.2, 78.0, 10.0])

exp1_data = exp1_sf([None], true_params) \
    + np.random.normal(0, 0.03, 100)
exp2_data = exp2_sf([None], true_params) \
    + np.random.normal(0, 0.025, 100)

run_poly(signal_poly_prior, jointlikelihood, 'test_joint', nlive=1000, RESUME=True)
run_poly(signal_poly_prior, exp1likelihood, 'test_exp1', nlive=1000, RESUME=True)
run_poly(signal_poly_prior, exp2likelihood, 'test_exp2', nlive=1000, RESUME=True)

exp1_samples = read_chains('test_exp1/test')
exp2_samples = read_chains('test_exp2/test')
joint_samples = read_chains('test_joint/test')

R_out_tension = joint_samples.logZ() - exp1_samples.logZ() - exp2_samples.logZ()

# In tension example...

exp1_data = exp1_sf([None], [0.2, 78.0, 10.0]) \
    + np.random.normal(0, 0.03, 100)
exp2_data = exp2_sf([None], [0.25, 84.0, 13.0]) \
    + np.random.normal(0, 0.025, 100)

run_poly(signal_poly_prior, jointlikelihood, 'test_joint_in_tension', nlive=1000)
run_poly(signal_poly_prior, exp1likelihood, 'test_exp1_in_tension', nlive=1000)
run_poly(signal_poly_prior, exp2likelihood, 'test_exp2_in_tension', nlive=1000)

exp1_samples = read_chains('test_exp1_in_tension/test')
exp2_samples = read_chains('test_exp2_in_tension/test')
joint_samples = read_chains('test_joint_in_tension/test')

R = joint_samples.logZ() - exp1_samples.logZ() - exp2_samples.logZ()
print(R_out_tension)
print(R)