import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from anesthetic import read_chains
from tensionnet.tensionnet import nre
from tensionnet.robs import run_poly
from pypolychord.priors import UniformPrior, LogUniformPrior
import os
from tensionnet.tensionnet import nre
from tensionnet.utils import calcualte_stats
import tensorflow as tf
from scipy.stats import ecdf
import matplotlib as mpl
from matplotlib import rc
from scipy.stats import norm

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

base = 'chains/21cm_different_priors/'
if not os.path.exists(base):
    os.mkdir(base)
RESUME = False
RETRAIN = True

exp1_freq = np.linspace(60, 90, 50)
exp2_freq = np.linspace(70, 120, 50)
exp1_sf = signal_func_gen(exp1_freq)
exp2_sf = signal_func_gen(exp2_freq)
exp1_sf_nre = nre_signal_func_gen(exp1_freq)
exp2_sf_nre = nre_signal_func_gen(exp2_freq)

true_params = np.array([0.25, 75.0, 10.0])

#wide_prior_bounds = np.array([[0., 8.0], [40.0, 100.0], [1.0, 50.0], [0.01, 1]])
#conservative_prior_bounds = np.array([[0.0, 4.0], [60.0, 90.0], [5.0, 40.0], [0.01, 0.5]])
#narrow_prior_bounds = np.array([[0.0, 2.0], [70.0, 85.0], [5.0, 20.0], [0.01, 0.1]])
wide_prior_bounds = np.array([[0., 0.5], [50.0, 100.0], [1.0, 19.0], [0.001, 0.05]])
conservative_prior_bounds = np.array([[0.1, 0.4], [60.0, 90.0], [3, 17.0], [0.01, 0.04]])
narrow_prior_bounds = np.array([[0.15, 0.35], [70.0, 80.0], [5.0, 15.0], [0.015, 0.035]])

prior_sets = [wide_prior_bounds, 
              conservative_prior_bounds,
              narrow_prior_bounds
              ]
prior_sets_names = ['wide', 
                    'conservative', 
                    'narrow'
                    ]
Rs = []

"""fig, axes = plt.subplots(3, 4, figsize=(6.3, 6.3), sharex='col')
for i, ps in enumerate(prior_sets):
    nre_signal_prior = build_nre_priors(ps)
    samples = nre_signal_prior(1000)
    ax = axes[i, :].flatten()
    for j in range(samples.shape[-1]):
        ax[j].hist(samples[:, j], bins=50, density=True)
plt.show()
sys.exit(1)"""

fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.3))
for i, ps in enumerate(prior_sets):
    signal_prior, joint_prior = build_priors(ps)
    sbase = base + prior_sets_names[i] + '/'
    if not os.path.exists(sbase):
        os.mkdir(sbase)

    if RESUME:
        exp1_data = np.loadtxt(sbase + 'exp1_data.txt')
    else:
        exp1_data = exp1_sf(true_params) \
            + np.random.normal(0, 0.025, len(exp1_freq))
        np.savetxt(sbase + 'exp1_data.txt', exp1_data)

    run_poly(signal_prior, exp1likelihood, sbase + f'exp1',
             nlive=100*2, RESUME=RESUME, nDims=4)
    exp1_samples = read_chains(sbase + f'exp1/test')

    if RESUME:
        exp2_data = np.loadtxt(sbase + f'exp2_data.txt')
    else:
        exp2_data = exp2_sf(true_params) \
            + np.random.normal(0, 0.025, len(exp2_freq))
        np.savetxt(sbase + f'exp2_data.txt', exp2_data)

    run_poly(joint_prior, jointlikelihood, sbase + f'joint',
             nlive=125*2, RESUME=RESUME, nDims=5)
    run_poly(signal_prior, exp2likelihood, sbase + f'exp2',
             nlive=100*2, RESUME=RESUME, nDims=4)

    exp2_samples = read_chains(sbase + f'exp2/test')
    joint_samples = read_chains(sbase + f'joint/test')


    Rs.append((joint_samples.logZ(1000) - 
              exp1_samples.logZ(1000) - exp2_samples.logZ(1000)).values)
    
    nre_signal_prior = build_nre_priors(ps)

    if not RETRAIN:
        nrei = nre.load(sbase + 'model_test.pkl',
                exp1_sf_nre, exp2_sf_nre, nre_signal_prior)
    else:
        nrei = nre(lr=1e-4)
        nrei.build_model(len(exp1_freq) + len(exp2_freq), 
                            [25]*5, 'sigmoid')#, skip_layers=False)
        nrei.build_simulations(exp1_sf_nre, exp2_sf_nre, 
                               nre_signal_prior, n=250000)
        model, data_test, labels_test = nrei.training(epochs=300, 
                                                      batch_size=124)
        nrei.save(sbase + 'model_test.pkl')

    examples = nrei.__call__(iters=5000)
    r = nrei.r_values
    mask = np.isfinite(r)

    Robs = Rs[-1].mean()
    errorRs = np.std(Rs[-1])

    axes[i, 0].hist(r[mask], bins=50, density=True)
    axes[i, 0].axvline(Robs, color='r', ls='--')
    axes[i, 0].set_title(r'$\log R_{obs}=$' + str(np.round(Robs, 2)) + r'$\pm$' +
                            str(np.round(errorRs, 2)))
    axes[i, 0].axvspan(Robs - errorRs, Robs + errorRs, alpha=0.1, color='r')

    #if i > 0:
    #    axes[i, 0].set_xlim(axes[0, 0].get_xlim()[0], 
    #                        axes[0, 0].get_xlim()[1])
    axes[i, 0].set_xlim(3, 9)

    r  = np.sort(r[mask])
    c = ecdf(r)


    sigmaD, sigma_D_upper, sigma_D_lower, \
            sigmaA, sigma_A_upper, sigma_A_lower, \
                sigmaR, sigmaR_upper, sigmaR_lower = \
                    calcualte_stats(Robs, errorRs, c)
    print('Prior Set:', prior_sets_names[i])
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
    np.savetxt(sbase + f'tension_stats.txt',
                np.hstack([sigmaD, sigma_D_upper, sigma_D_lower,
                            sigmaA, sigma_A_upper, sigma_A_lower,
                            sigmaR, sigmaR_upper, sigmaR_lower]).T)
    
    axes[i, 1].plot(r, c.cdf.evaluate(r))
    axes[i, 1].axhline(c.cdf.evaluate(Robs), ls='--',
                color='r')
    axes[i, 1].set_title(r'$\sigma_D =$' + f'{sigmaD:.3f}' + 
                         r'$+$' + f'{np.abs(sigmaD - sigma_D_upper):.3f}' +
                r'$(-$' + f'{np.abs(sigma_D_lower - sigmaD):.3f}' + r'$)$' + '\n' +
                r'$\sigma_A=$' + f'{sigmaA:.3f}' + 
                r'$+$' + f'{np.abs(sigmaA - sigma_A_upper):.3f}' +
                r'$(-$' + f'{np.abs(sigma_A_lower - sigmaA):.3f}' + r'$)$')
    axes[i, 1].axhspan(c.cdf.evaluate(Robs - errorRs), 
            c.cdf.evaluate(Robs + errorRs), 
            alpha=0.1, 
            color='r')

    if i == 0:
        prior_label = 'Wide'
        axes[i, 0].set_ylabel('Wide Prior\nDensity')
        axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')
    elif i == 1:
        prior_label = 'Conservative'
        axes[i, 0].set_ylabel('Conservative Prior\nDensity')
        axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')
    else:
        prior_label = 'Narrow'
        axes[i, 0].set_ylabel('Narrow Prior\nDensity')
        axes[i, 1].set_ylabel(r'$P(\log R < \log R^\prime)$')
        axes[i, 0].set_xlabel(r'$\log R$')
        axes[i, 1].set_xlabel(r'$\log R$')

    
    axes[i, 2].axis('off')
    axes[i, 2].table(cellText=[[str(ps[0, 0]) + ' - ' + str(ps[0, 1])],
                                [str(ps[1, 0]) + ' - ' + str(ps[1, 1])], 
                                [str(ps[2, 0]) + ' - ' + str(ps[2, 1])],
                                [str(ps[3, 0]) + ' - ' + str(ps[3, 1])]],
                     colLabels=[prior_label],
                     rowLabels=[r'$A$', r'$\nu_0$', r'$w$', r'$\sigma$'],
                     cellLoc='center',
                     loc='center',
                     fontsize=15)

plt.tight_layout()
plt.savefig('figures/figure5.pdf', bbox_inches='tight')
plt.close()

