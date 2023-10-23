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

def joint_prior(cube):
    theta = np.zeros(5)
    theta[0] = UniformPrior(0, 4)(cube[0]) #amp
    theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
    theta[2] = UniformPrior(5, 40)(cube[2]) #w
    theta[3] = LogUniformPrior(0.001, 0.1)(cube[3]) #exp1 noise
    theta[4] = LogUniformPrior(0.001, 0.1)(cube[4]) #exp2 noise
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
    exp1theta = theta[:4]
    exp2theta = np.concatenate((theta[:3], theta[4:]))
    return exp1likelihood(exp1theta)[0] + exp2likelihood(exp2theta)[0], []

base = 'test_case_chains/'
RESUME = False

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)

true_params = np.array([0.2, 78.0, 10.0])

try:
    exp1_data = np.loadtxt(base + 'exp1_data_no_tension.txt')
    exp2_data = np.loadtxt(base + 'exp2_data_no_tension.txt')
except:
    exp1_data = exp1_sf([None], true_params) \
        + np.random.normal(0, 0.005, 100)
    exp2_data = exp2_sf([None], true_params) \
        + np.random.normal(0, 0.005, 100)
    np.savetxt(base + 'exp1_data_no_tension.txt', exp1_data)
    np.savetxt(base + 'exp2_data_no_tension.txt', exp2_data)

#run_poly(joint_prior, jointlikelihood, base + 'test_joint', nlive=1000, RESUME=RESUME, nDims=5)
#run_poly(signal_poly_prior, exp1likelihood, base + 'test_exp1', nlive=1000, RESUME=RESUME)
#run_poly(signal_poly_prior, exp2likelihood, base + 'test_exp2', nlive=1000, RESUME=RESUME)

exp1_samples = read_chains(base + 'test_exp1/test')
exp2_samples = read_chains(base + 'test_exp2/test')
joint_samples = read_chains(base + 'test_joint/test')

R_out_tension = joint_samples.logZ(1000) - exp1_samples.logZ(1000) - exp2_samples.logZ(1000)

# In tension example...

try:
    exp1_data = np.loadtxt(base + 'exp1_data_in_tension.txt')
    exp2_data = np.loadtxt(base + 'exp2_data_in_tension.txt')
except:
    exp1_data = exp1_sf([None], [0.2, 78.0, 10.0]) \
        + np.random.normal(0, 0.005, 100)
    exp2_data = exp2_sf([None], [0.25, 82.0, 12.0]) \
        + np.random.normal(0, 0.005, 100)
    np.savetxt(base + 'exp1_data_in_tension.txt', exp1_data)
    np.savetxt(base + 'exp2_data_in_tension.txt', exp2_data)

#run_poly(joint_prior, jointlikelihood, base + 'test_joint_in_tension', nlive=1000, RESUME=RESUME, nDims=5)
#run_poly(signal_poly_prior, exp1likelihood, base + 'test_exp1_in_tension', nlive=1000, RESUME=RESUME)
#run_poly(signal_poly_prior, exp2likelihood, base + 'test_exp2_in_tension', nlive=1000, RESUME=RESUME)

exp1_samples = read_chains(base + 'test_exp1_in_tension/test')
exp2_samples = read_chains(base + 'test_exp2_in_tension/test')
joint_samples = read_chains(base + 'test_joint_in_tension/test')

R = joint_samples.logZ(1000) - exp1_samples.logZ(1000) - exp2_samples.logZ(1000)
print('No Tension: ', R_out_tension.mean(), R_out_tension.std())
print('Tension: ', R.mean(), R.std())

# very consistent test case...

try:
    exp1_data = np.loadtxt(base + 'exp1_data_consistent.txt')
    exp2_data = np.loadtxt(base + 'exp2_data_consistent.txt')
except:
    exp1_data = exp1_sf([None], true_params) \
        + np.random.normal(0, 0.005, 100)
    exp2_data = exp2_sf([None], true_params) \
        + np.random.normal(0, 0.005, 100)
    
    exp1_data[exp1_freq >= exp2_freq.min()] = \
        exp1_sf([None], true_params)[exp1_freq >= exp2_freq.min()]
    exp2_data[exp2_freq <= exp1_freq.max()] = \
        exp2_sf([None], true_params)[exp2_freq <= exp1_freq.max()]
    
    np.savetxt(base + 'exp1_data_consistent.txt', exp1_data)
    np.savetxt(base + 'exp2_data_consistent.txt', exp2_data)

run_poly(joint_prior, jointlikelihood, base + 'test_joint_consistent', nlive=1000, RESUME=RESUME, nDims=5)
run_poly(signal_poly_prior, exp1likelihood, base + 'test_exp1_consistent', nlive=1000, RESUME=RESUME)
run_poly(signal_poly_prior, exp2likelihood, base + 'test_exp2_consistent', nlive=1000, RESUME=RESUME)

exp1_samples = read_chains(base + 'test_exp1_consistent/test')
exp2_samples = read_chains(base + 'test_exp2_consistent/test')
joint_samples = read_chains(base + 'test_joint_consistent/test')

R = joint_samples.logZ(1000) - exp1_samples.logZ(1000) - exp2_samples.logZ(1000)
print('Very Conistent: ', R.mean(), R.std())