import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings
from sklearn.model_selection import train_test_split
from tensionnet.utils import plotting_preamble
from tqdm import tqdm
from anesthetic import read_chains
import pypolychord
from scipy.stats import poisson
import os

plotting_preamble()

base_dir = 'particle_physics_example/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

np.random.seed(1420)

def prior_individual(hypercube):
    theta = np.zeros(9)
    theta[0] = UniformPrior(0, 10)(hypercube[0])
    theta[1] = UniformPrior(0, 5)(hypercube[1])
    theta[2] = UniformPrior(0, 10)(hypercube[2])
    theta[3] = UniformPrior(0, 5)(hypercube[3])
    theta[4] = UniformPrior(0, 10)(hypercube[4])
    theta[5] = UniformPrior(0, 5)(hypercube[5])

    theta[6] = UniformPrior(0, 1000)(hypercube[6])
    theta[7] = UniformPrior(100, 180)(hypercube[7])
    theta[8] = UniformPrior(0, 10)(hypercube[8])
    return theta

def background_model(x, theta):
    amplitude = theta[::2]
    exponent = theta[1::2]
    return np.sum([a*np.exp(-b*x) for a, b in zip(amplitude, exponent)], axis=0)

def signal_model(x, phi):
    amplitude = phi[0]
    mean = phi[1]
    sigma = phi[2]
    return amplitude * np.exp(-0.5 * (x - mean)**2 / sigma**2)


###############################################################################
########################## set up mock data ###################################
###############################################################################

length = 100
x = np.linspace(100, 180, length)
normx = (x - np.max(x))/(np.max(x) - np.min(x))

fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
truebgparams = prior_individual(np.random.uniform(0, 1, 9))[:6]
theorybg = background_model(normx, truebgparams)
theorysig1 = signal_model(x, [0.12*theorybg.max(), 123, 5])
theory1 = theorybg + theorysig1
data1 = poisson.rvs(theory1, size=length)

axes[0].scatter(x, data1)
axes[0].plot(x, theory1)
axes[0].axvline(123, color='red', linestyle='--')
axes[0].set_title('Weaker excess at 123 GeV')
axes[1].scatter(x, data1 - theorybg)
axes[1].axhline(0, color='black', linestyle='--')
axes[1].plot(x, theorysig1)    
axes[1].set_title('Residuals')

theorysig2 = signal_model(x, [0.15*theorybg.max(), 125, 3])
theory2 = theorybg + theorysig2
data2 = poisson.rvs(theory2, size=length)
axes[2].scatter(x, data2, color='red')
axes[2].plot(x, theory2, color='red')
axes[2].axvline(125, color='red', linestyle='--')
axes[2].set_title('Stronger excess at 125 GeV')
axes[3].scatter(x, data2 - theorybg, color='red')
axes[3].plot(x, theorysig2, color='red')
axes[3].set_title('Residuals')
axes[3].axhline(0, color='black', linestyle='--')

axes[3].set_xlabel('Mass [GeV]')

for i in range(4):
    axes[i].set_ylabel('Events')
plt.tight_layout()
plt.savefig(base_dir + 'data.png')
plt.close()
#plt.show()

jointnames = [('b%i' % i, r'b_%i' % i) for i in range(6)] + \
    [('\mu', r'\mu'), ('A1', r'A_1'), ('sigma1', r'\sigma_1')] + \
    [('A2', r'A_2'), ('sigma2', r'\sigma_2')]

exp1names = [('b%i' % i, r'b_%i' % i)  for i in range(6)] + \
    [('A1', r'A_1'), ('\mu', r'\mu'), ('sigma1', r'\sigma_1')]

exp2names = [('b%i' % i, r'b_%i' % i)  for i in range(6)] + \
    [('A2', r'A_2'), ('\mu', r'\mu'), ('sigma2', r'\sigma_2')]

skip_poly = True

if not skip_poly:
    ###############################################################################
    ################################ NS on exp1 ###################################
    ###############################################################################

    def likelihood1(params):
        model = background_model(normx, params[:6]) + signal_model(x, params[6:])
        return poisson.logpmf(data1, model).sum(), []

    nDims = 9
    nDerived = 0

    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir = base_dir + 'exp1/'

    output = pypolychord.run_polychord(likelihood1, nDims, 
                                        nDerived, settings, prior_individual)
    paramnames = exp1names
    output.make_paramnames_files(paramnames)

    ###############################################################################
    ################################ NS on exp2 ###################################
    ###############################################################################

    def likelihood2(params):
        model = background_model(normx, params[:6]) + signal_model(x, params[6:])
        return poisson.logpmf(data2, model).sum(), []

    nDims = 9

    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir = base_dir + 'exp2/'

    output = pypolychord.run_polychord(likelihood2, nDims, 
                                        nDerived, settings, prior_individual)
    paramnames = exp2names
    output.make_paramnames_files(paramnames)

    ###############################################################################
    ################################ NS on joint ###################################
    ###############################################################################

    def prior_joint(hypercube):
        theta = np.zeros(11)
        theta[0] = UniformPrior(0, 10)(hypercube[0])
        theta[1] = UniformPrior(0, 5)(hypercube[1])
        theta[2] = UniformPrior(0, 10)(hypercube[2])
        theta[3] = UniformPrior(0, 5)(hypercube[3])
        theta[4] = UniformPrior(0, 10)(hypercube[4])
        theta[5] = UniformPrior(0, 5)(hypercube[5])

        theta[6] = UniformPrior(100, 180)(hypercube[6])

        theta[7] = UniformPrior(0, 1000)(hypercube[7])
        theta[8] = UniformPrior(0, 10)(hypercube[8])

        theta[9] = UniformPrior(0, 1000)(hypercube[9])
        theta[10] = UniformPrior(0, 10)(hypercube[10])

        return theta

    def likelihoodjoint(params):
        bgparams = params[:6]
        sig1params = [params[7], params[6], params[8]]
        sig2params = [params[9], params[6], params[10]]
        like1, [] = likelihood1(np.concatenate([bgparams, sig1params]))
        like2, [] = likelihood2(np.concatenate([bgparams, sig2params]))
        return like1+like2, []

    nDims = 11

    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = False
    settings.base_dir = base_dir + 'joint/'

    output = pypolychord.run_polychord(likelihoodjoint, nDims, 
                                        nDerived, settings, prior_joint)
    paramnames = jointnames
    output.make_paramnames_files(paramnames)

###############################################################################
############################# calculate R #####################################
###############################################################################


chainsj = read_chains(base_dir + 'joint/test', columns=[jn[0] for jn in jointnames])
axes = chainsj.plot_2d(['\mu'])

chains1 = read_chains(base_dir + 'exp1/test', columns=[e1n[0] for e1n in exp1names])
chains1.plot_2d(axes)

chains2 = read_chains(base_dir + 'exp2/test', columns=[e2n[0] for e2n in exp2names])
chains2.plot_2d(axes)

#plt.show()
plt.savefig(base_dir + 'chains.pdf', bbox_inches='tight')
plt.close()

jointevidence = chainsj.logZ(10000)
exp1evidence = chains1.logZ(10000)
exp2evidence = chains2.logZ(10000)

Rs = (jointevidence - exp1evidence - exp2evidence)
R = Rs.mean()
errorR = Rs.std()
print('R = %.2f +/- %.2f' % (R, errorR))

##############################################################################
################################ Do NRE ######################################
##############################################################################

print('Running NRE...')

import random

nSamples = 100000
load_data = False

prior_mins = [0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0]
prior_maxs = [10, 5, 10, 5, 10, 5, 180, 1000, 10, 1000, 10]

def nre_prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(11)]).T

def simulation(theta):

    exp1s, exp2s = [], []
    for i in tqdm(range(len(theta))):
        bgparams = theta[i, :6]
        sig1params = [theta[i, 7], theta[i, 6], theta[i, 8]]
        sig2params = [theta[i, 9], theta[i, 6], theta[i, 10]]
        exp1s.append(poisson.rvs(background_model(normx, bgparams) +
                                    signal_model(x, sig1params), size=len(x)))
        exp2s.append(poisson.rvs(background_model(normx, bgparams) +
                                    signal_model(x, sig2params), size=len(x)))
    exp1s = np.array(exp1s)
    exp2s = np.array(exp2s)

    idx = np.arange(len(exp1s))
    np.random.shuffle(idx)
    
    stacked = np.hstack([exp1s, exp2s, np.array([[1]*len(exp1s)]).T])
    shuffled = np.hstack([exp1s, exp2s[idx], np.array([[0]*len(exp1s)]).T])

    # validation set
    idx = random.sample(range(len(stacked)), int(0.1*len(stacked)))
    data_validation = stacked[idx, :-1]
    stacked = np.delete(stacked, idx, axis=0)
    shuffled = np.delete(shuffled, idx, axis=0)

    data =  np.concatenate([stacked, shuffled])

    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]

    labels = data[:, -1]
    data = data[:, :-1]

    data_train, data_test, labels_train, labels_test = \
                train_test_split(data, labels, 
                                 test_size=0.33)
        
    labels_test = labels_test
    labels_train = labels_train

    data_test = (data_test - data_train.mean(axis=0)) / \
        data_train.std(axis=0)
    data_validation = (data_validation - data_train.mean(axis=0)) / \
        data_train.std(axis=0)
    data_train = (data_train - data_train.mean(axis=0)) / \
        data_train.std(axis=0)
    
    return data_train, data_test, data_validation, labels_train, labels_test

from tensionnet.tensionnet import nre
from scipy.stats import ecdf
from tensionnet.utils import calcualte_stats
from tensorflow.keras.optimizers.schedules import ExponentialDecay  

load_trained_nre = False

if load_data:
    data_train = np.load(base_dir + 'data_train.npy')
    data_test = np.load(base_dir + 'data_test.npy')
    data_validation = np.load(base_dir + 'data_validation.npy')
    labels_train = np.load(base_dir + 'labels_train.npy')
    labels_test = np.load(base_dir + 'labels_test.npy')
else:
    data_train, data_test, data_validation, labels_train, labels_test = simulation(nre_prior(nSamples))
    np.save(base_dir + 'data_train.npy', data_train)
    np.save(base_dir + 'data_test.npy', data_test)
    np.save(base_dir + 'data_validation.npy', data_validation)
    np.save(base_dir + 'labels_train.npy', labels_train)
    np.save(base_dir + 'labels_test.npy', labels_test)

sigmaD, sigmaA = [], []
for i in range(5):
    
    if load_trained_nre:
        nrei = nre.load(base_dir + 'nre_run' + str(i) + '.pkl',
              None, None, nre_prior)
    else:
        lr = ExponentialDecay(1e-3, 1000, 0.9)
        #lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, 1000, warmup_target=1e-1, warmup_steps=1000)
        nrei = nre(lr=lr)
        nrei.build_model(200, [200]*2, 'sigmoid')
            
        nrei.data_train = data_train
        nrei.data_test = data_test
        nrei.labels_train = labels_train
        nrei.labels_test = labels_test
        nrei.simulation_func_A = None
        nrei.simulation_func_B = None
        nrei.shared_prior = nre_prior
        nrei.prior_function_A = None
        nrei.prior_function_B = None

        nrei.training(epochs=10, batch_size=1000, patience=5)
        nrei.save(base_dir + 'nre_run' + str(i) + '.pkl')

    plt.plot(nrei.loss_history, label='Training Loss')
    plt.plot(nrei.test_loss_history, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(base_dir + 'loss_run' + str(i) + '.pdf', bbox_inches='tight')
    #plt.show()
    plt.close()


    nrei.__call__(iters=data_validation[:1000])
    r = nrei.r_values
    mask = np.isfinite(r)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3))
    axes[0].hist(r[mask], bins=25, density=True)
    axes[0].set_xlabel(r'$\log R$')
    axes[0].set_ylabel('Density')
    axes[0].axvline(R, ls='--', c='r')
    axes[0].axvspan(R - errorR, R + errorR, alpha=0.1, color='r')

    rsort  = np.sort(r[mask])
    c = ecdf(rsort)

    axes[1].plot(rsort, c.cdf.evaluate(rsort)) 
    axes[1].axhline(c.cdf.evaluate(R), ls='--',
            color='r')
    axes[1].axhspan(c.cdf.evaluate(R - errorR), 
            c.cdf.evaluate(R + errorR), 
            alpha=0.1, 
            color='r')
    axes[1].set_xlabel(r'$\log R$')
    axes[1].set_ylabel(r'$P(\log R < \log R^\prime)$')

    axes[1].axhline(c.cdf.evaluate(R), ls='--',
                color='r')
    axes[1].axhspan(c.cdf.evaluate(R - errorR),
                c.cdf.evaluate(R + errorR),
                alpha=0.1,
                color='r')

    stats = calcualte_stats(R, errorR, c)
    print(stats)
    sigmaD.append(stats[:3])
    sigmaA.append(stats[3:6])

    plt.tight_layout()
    plt.savefig(base_dir + 'run' + str(i) + '.pdf', bbox_inches='tight')
    plt.close()

sigmaA = np.array(sigmaA)
sigmaD = np.array(sigmaD)

mean_sigmaA = sigmaA[:, 0].mean()
mean_sigmaD = sigmaD[:, 0].mean()

sigmaAs = sigmaA[:, 0]
sigmaA_lower = sigmaA[:, 0] - sigmaA[:, 2]
sigmaA_upper = sigmaA[:, 1] - sigmaA[:, 0]
sigmaDs = sigmaD[:, 0]
sigmaD_lower = sigmaD[:, 0] - sigmaD[:, 1]
sigmaD_upper = sigmaD[:, 2] - sigmaD[:, 0]

norm_sigmaA = sigmaAs / mean_sigmaA
norm_sigmaD = sigmaDs / mean_sigmaD

fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))
axes.errorbar(np.arange(5), sigmaAs, yerr=[sigmaA_lower, sigmaA_upper], fmt='o')
axes.set_xticks(np.arange(5))
axes.set_ylabel(r'$C$')
axes.set_xlabel('Run')

lower_mean_sigmaA_error = 1/np.sqrt(5) * np.sqrt(np.sum((sigmaA_lower)**2))
upper_mean_sigmaA_error = 1/np.sqrt(5) * np.sqrt(np.sum((sigmaA_upper)**2))
print(mean_sigmaA, lower_mean_sigmaA_error, upper_mean_sigmaA_error)


#for i in range(2):
axes.axhline(mean_sigmaA, ls='--', c='r')
axes.axhspan(mean_sigmaA - lower_mean_sigmaA_error, mean_sigmaA + upper_mean_sigmaA_error, alpha=0.1, color='r')
plt.tight_layout()
plt.savefig(base_dir + 'sigma.pdf', bbox_inches='tight')
plt.close()