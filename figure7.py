import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from anesthetic import read_chains
from tensionnet.tensionnet import nre
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import os
from tensionnet.tensionnet import nre
import tensorflow as tf
from scipy.stats import ecdf
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

def signal_func_gen(freqs):
    def signal(parameters):
        amp, nu_0, w = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2))
    return signal

def nre_signal_func_gen(freqs):
    def signal(parameters):
        amp, nu_0, w, sigma = parameters
        return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2)) + \
            np.random.normal(0, sigma, len(freqs))
    return signal

def exp1likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[-1]**2) \
        - 0.5 * (exp1_data - exp1_sf(theta[:-1]))**2/theta[-1]**2).sum(),[]

def exp2likelihood(theta):
    # gaussian log-likelihood
    return (-0.5 * np.log(2*np.pi*theta[-1]**2) \
        - 0.5 * (exp2_data - exp2_sf(theta[:-1]))**2/theta[-1]**2).sum(),[]

def jointlikelihood(theta):
    return exp1likelihood([*theta[:-2], theta[-2]])[0] + \
        exp2likelihood([*theta[:-2], theta[-1]])[0], []

def build_priors(prior_bounds):
    def signal_prior(cube):
        theta = np.zeros(4)
        theta[0] = UniformPrior(prior_bounds[0][0], prior_bounds[0][1])(cube[0]) #amp
        theta[1] = UniformPrior(prior_bounds[1][0], prior_bounds[1][1])(cube[1]) #nu_0
        theta[2] = UniformPrior(prior_bounds[2][0], prior_bounds[2][1])(cube[2]) #w
        theta[3] = UniformPrior(prior_bounds[3][0], prior_bounds[3][1])(cube[3])
        return theta

    def joint_prior(cube):
        theta = np.zeros(5)
        theta[0] = UniformPrior(prior_bounds[0][0], prior_bounds[0][1])(cube[0]) #amp
        theta[1] = UniformPrior(prior_bounds[1][0], prior_bounds[1][1])(cube[1]) #nu_0
        theta[2] = UniformPrior(prior_bounds[2][0], prior_bounds[2][1])(cube[2]) #w
        theta[3] = UniformPrior(prior_bounds[3][0], prior_bounds[3][1])(cube[3])
        theta[4] = UniformPrior(prior_bounds[3][0], prior_bounds[3][1])(cube[4])
        return theta
    return signal_prior, joint_prior

def build_nre_priors(prior_bounds):
    def signal_prior(n):
        parameters = np.ones((n, 4))
        parameters[:, 0] = np.random.uniform(prior_bounds[0][0], prior_bounds[0][1], n) #amp
        parameters[:, 1] = np.random.uniform(prior_bounds[1][0], prior_bounds[1][1], n) #nu_0
        parameters[:, 2] = np.random.uniform(prior_bounds[2][0], prior_bounds[2][1], n) #w
        parameters[:, 3] = np.random.uniform(prior_bounds[3][0], prior_bounds[3][1], n) #sigma
        return parameters
    return signal_prior


base = 'chains/21cm_direct_prediction/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME = False

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)
exp1_sf_nre = nre_signal_func_gen(exp1_freq)
exp2_sf_nre = nre_signal_func_gen(exp2_freq)

true_params = np.array([0.2, 78.0, 10.0])

prior_bounds = np.array([[0.0, 4.0], [60.0, 90.0], [5.0, 40.0], [0.001, 0.1]])

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3))
signal_prior, joint_prior = build_priors(prior_bounds)

if not os.path.exists(base):
    os.mkdir(base)

######################################################################
########## Generate data and do fits #################################
######################################################################
try:
    exp1_data = np.loadtxt(base + 'exp1_data.txt')
except:
    exp1_data = exp1_sf(true_params) \
        + np.random.normal(0, 0.025, 100) \
        + (exp1_freq/78)**(-2.5)*0.050*np.sin(2*np.pi*exp1_freq/5 + 0.5)
    np.savetxt(base + 'exp1_data.txt', exp1_data)

axes[2].plot(exp1_freq, exp1_data - exp1_sf(true_params) 
             - (exp1_freq/78)**(-2.5)*0.050*np.sin(2*np.pi*exp1_freq/5 + 0.5))
axes[2].plot(exp1_freq, (exp1_freq/78)**(-2.5)*0.050*np.sin(2*np.pi*exp1_freq/5 + 0.5), 
             color='C2')
axes[2].set_title('Systematic')
axes[2].set_xlabel('Frequency [MHz]')
axes[2].set_ylabel(r'$\delta T_b$ [K]')

run_poly(signal_prior, exp1likelihood, base + f'exp1',
            nlive=100, RESUME=RESUME, nDims=4)
exp1_samples = read_chains(base + f'exp1/test')

try:
    exp2_data = np.loadtxt(base + f'exp2_data.txt')
except:
    exp2_data = exp2_sf(true_params) \
        + np.random.normal(0, 0.025, 100)
    np.savetxt(base + f'exp2_data.txt', exp2_data)

run_poly(joint_prior, jointlikelihood, base + f'joint',
            nlive=1250, RESUME=RESUME, nDims=5)
run_poly(signal_prior, exp2likelihood, base + f'exp2',
            nlive=100, RESUME=RESUME, nDims=4)

######################################################################
#################### Read chains and calcualte R #####################
######################################################################
exp2_samples = read_chains(base + f'exp2/test')
joint_samples = read_chains(base + f'joint/test')

Rs = (joint_samples.logZ(1000) - 
            exp1_samples.logZ(1000) - exp2_samples.logZ(1000)).values


######################################################################
######### Load NRE trained without systematics #######################
######################################################################
nre_signal_prior = build_nre_priors(prior_bounds)
try:
    nrei = nre.load('figure7-nre.pkl',
                exp2_sf_nre, exp1_sf_nre, nre_signal_prior)
except:
    nrei = nre(lr=1e-4)
    nrei.build_model(len(exp2_freq) + len(exp1_freq),
                        [25]*5, 'sigmoid')
    nrei.build_simulations(exp2_sf_nre, exp1_sf_nre, nre_signal_prior, n=200000)
    model, data_test, labels_test = nrei.training(epochs=1000, 
                                                  batch_size=1000)
    nrei.save('figure7-nre.pkl')

######################################################################
######### NRE R distribution #########################################
######################################################################
nrei.__call__(iters=5000)
r = nrei.r_values
mask = np.isfinite(r)

Robs = Rs.mean()
errorRs = np.std(Rs)

axes[0].hist(r[mask], bins=50, density=True)
axes[0].axvline(Robs, color='r', ls='--', label='Nested Sampling')
axes[0].set_title(r'$\log R_{obs}=$' + str(np.round(Robs, 2)) + r'$\pm$' +
                        str(np.round(errorRs, 2)))
axes[0].axvspan(Robs - errorRs, Robs + errorRs, alpha=0.1, color='r')

r  = np.sort(r[mask])
c = ecdf(r)
axes[1].plot(r, c.cdf.evaluate(r))
axes[1].axhline(c.cdf.evaluate(Robs), ls='--',
            color='r')

from tensionnet.utils import calcualte_stats

sigmaD, sigma_D_upper, sigma_D_lower, \
            sigmaA, sigma_A_upper, sigma_A_lower, \
                sigmaR, sigmaR_upper, sigmaR_lower = \
                    calcualte_stats(Robs, errorRs, c)
print(f'Rs: {Robs}, Rs_upper: {Robs + errorRs},' + 
        f'Rs_lower: {Robs - errorRs}')
print(f'sigmaD: {sigmaD}, sigma_D_upper: ' + 
        f'{np.abs(sigmaD - sigma_D_upper)}, ' +
        f'sigma_D_lower: {np.abs(sigma_D_lower - sigmaD)}')
print(f'sigmaA: {sigmaA}, sigma_A_upper: ' +
        f'{np.abs(sigmaA - sigma_A_upper)}, ' +
        f'sigma_A_lower: {np.abs(sigma_A_lower - sigmaA)}')
print(f'sigmaR: {sigmaR}, sigmaR_upper: ' + 
        f'{np.abs(sigmaR - sigmaR_upper)}, ' +
        f'sigmaR_lower: {np.abs(sigmaR_lower - sigmaR)}')
np.savetxt(base + f'tension_stats.txt',
            np.hstack([sigmaD, sigma_D_upper, sigma_D_lower,
                        sigmaA, sigma_A_upper, sigma_A_lower,
                        sigmaR, sigmaR_upper, sigmaR_lower]).T)

"""axes[1].set_title(r'$\sigma_D =$' + f'{sigmaD:.3f}' + r'$+$' + 
                  f'{np.abs(sigmaD - sigma_D_upper):.3f}' +
            r'$(-$' + f'{np.abs(sigma_D_lower - sigmaD):.3f}' + 
            r'$)$' + '\n' +
            r'$\sigma_A =$' + f'{sigmaA:.3f}' + r'$+$' +
            f'{np.abs(sigmaA - sigma_A_upper):.3f}' +
            r'$(-$' + f'{np.abs(sigma_A_lower - sigmaA):.3f}')"""
axes[1].axhspan(c.cdf.evaluate(Robs - errorRs), 
        c.cdf.evaluate(Robs + errorRs), 
        alpha=0.1, 
        color='r')

######################################################################
########### What if put raw data into the network? ####################
######################################################################
nrei.__call__(iters=np.array([np.concatenate([exp1_data, exp2_data])]))
r = nrei.r_values

from scipy.stats import norm

sigmaD = norm.isf(c.cdf.evaluate(r)/2)
p = 2 - 2*c.cdf.evaluate(r)
sigmaR = norm.isf(p/2)
sigmaA = norm.isf((1-c.cdf.evaluate(r))/2)
print('Direct Prediction: ')
print(f'Rs: {r}')
print(f'CDF(Rs): {c.cdf.evaluate(r)}')
print(f'sigmaD: {sigmaD}')
print(f'sigmaA: {sigmaA}')
print(f'sigmaR: {sigmaR}')

axes[0].axvline(r, color='purple', ls='--', label='Direct Prediction')
axes[1].axhline(c.cdf.evaluate(r), ls='--',
            color='purple')


axes[0].set_ylabel('Density')
axes[1].set_ylabel(r'$P(\log R < \log R^\prime)$')
axes[0].set_xlabel(r'$\log R$')
axes[1].set_xlabel(r'$\log R$')


plt.tight_layout()
plt.savefig('figures/figure7.pdf', bbox_inches='tight')
plt.close()

