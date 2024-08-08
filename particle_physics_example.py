import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior, LogUniformPrior
from pypolychord.settings import PolyChordSettings
from anesthetic.plot import kde_plot_1d
from sklearn.model_selection import train_test_split
from tensionnet.utils import plotting_preamble
from tqdm import tqdm
from anesthetic import read_chains
import pypolychord
from scipy.stats import poisson
import os

plotting_preamble()

base_dir = 'particle_physics_example_120_120_diff_data/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

#r = np.random.randint(0, 1000)
#print(r)
np.random.seed(774)


def prior_individual(hypercube):
    theta = np.zeros_like(hypercube)
    theta[0] = UniformPrior(5, 10)(hypercube[0])
    theta[1] = UniformPrior(1, 8)(hypercube[1])
    theta[2] = UniformPrior(5, 10)(hypercube[2])
    theta[3] = UniformPrior(1, 8)(hypercube[3])
    theta[4] = UniformPrior(5, 10)(hypercube[4])
    theta[5] = UniformPrior(1, 8)(hypercube[5])

    theta[6] = UniformPrior(0, 800)(hypercube[6])
    theta[7] = UniformPrior(110, 150)(hypercube[7])
    theta[8] = UniformPrior(0.1, 10)(hypercube[8])
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
truebgparams1 = prior_individual(np.random.uniform(0, 1, 9))[:6]
theorybg1 = background_model(normx, truebgparams1)
theorysig1 = signal_model(x, [0.12*theorybg1.max(), 125.1, 1.5])
theory1 = theorybg1 + theorysig1
data1 = poisson.rvs(theory1, size=length)

axes[0].scatter(x, data1)
axes[0].plot(x, theory1)
for i in range(4):
    axes[i].axvline(125.1, color='red', linestyle='--')
axes[0].set_title('Stronger excess')
axes[1].scatter(x, data1 - theorybg1)
axes[1].axhline(0, color='black', linestyle='--')
axes[1].plot(x, theorysig1)    
axes[1].set_title('Residuals')

truebgparams2 = prior_individual(np.random.uniform(0, 1, 9))[:6]
theorybg2 = background_model(normx, truebgparams2)

theorysig2 = signal_model(x, [0.15*theorybg2.max(), 125.1, 2])
theory2 = theorybg2 + theorysig2
data2 = poisson.rvs(theory2, size=length)
axes[2].scatter(x, data2, color='red')
axes[2].plot(x, theory2, color='red')
#axes[2].axvline(125.1, color='red', linestyle='--')
axes[2].set_title('Weaker excess')
axes[3].scatter(x, data2 - theorybg2, color='red')
axes[3].plot(x, theorysig2, color='red')
axes[3].set_title('Residuals')
axes[3].axhline(0, color='black', linestyle='--')

axes[3].set_xlabel('Mass [GeV]')

for i in range(4):
    axes[i].set_ylabel('Events')
plt.tight_layout()
plt.savefig(base_dir + 'data.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
#exit()

jointnames = [('b%i' % i, r'b_%i' % i) for i in range(12)] + \
    [('\mu', r'\mu'), ('A1', r'A_1'), ('sigma1', r'\sigma_1')] + \
    [('A2', r'A_2'), ('sigma2', r'\sigma_2')]

exp1names = [('b%i' % i, r'b_%i' % i)  for i in range(6)] + \
    [('A1', r'A_1'), ('\mu', r'\mu'), ('sigma1', r'\sigma_1')]

exp2names = [('b%i' % i, r'b_%i' % i)  for i in range(6)] + \
    [('A2', r'A_2'), ('\mu', r'\mu'), ('sigma2', r'\sigma_2')]

skip_poly = False

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
        theta = np.zeros_like(hypercube)
        theta[0] = UniformPrior(5, 10)(hypercube[0])
        theta[1] = UniformPrior(1, 8)(hypercube[1])
        theta[2] = UniformPrior(5, 10)(hypercube[2])
        theta[3] = UniformPrior(1, 8)(hypercube[3])
        theta[4] = UniformPrior(5, 10)(hypercube[4])
        theta[5] = UniformPrior(1, 8)(hypercube[5])

        theta[6] = UniformPrior(5, 10)(hypercube[6])
        theta[7] = UniformPrior(1, 8)(hypercube[7])
        theta[8] = UniformPrior(5, 10)(hypercube[8])
        theta[9] = UniformPrior(1, 8)(hypercube[9])
        theta[10] = UniformPrior(5, 10)(hypercube[10])
        theta[11] = UniformPrior(1, 8)(hypercube[11])

        theta[12] = UniformPrior(110, 150)(hypercube[12])

        theta[13] = UniformPrior(0, 800)(hypercube[13])
        theta[14] = UniformPrior(0.1, 10)(hypercube[14])

        theta[15] = UniformPrior(0, 800)(hypercube[15])
        theta[16] = UniformPrior(0.1, 10)(hypercube[16])

        return theta

    def likelihoodjoint(params):
        bgparams1 = params[:6]
        bgparams2 = params[6:12]
        sig1params = [params[13], params[12], params[14]]
        sig2params = [params[15], params[12], params[16]]
        like1, [] = likelihood1(np.concatenate([bgparams1, sig1params]))
        like2, [] = likelihood2(np.concatenate([bgparams2, sig2params]))
        return like1+like2, []

    nDims = 17

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
axes = chainsj.plot_2d(['\mu'], figsize=(5, 4.5))

chains1 = read_chains(base_dir + 'exp1/test', columns=[e1n[0] for e1n in exp1names])
chains1.plot_2d(axes)

chains2 = read_chains(base_dir + 'exp2/test', columns=[e2n[0] for e2n in exp2names])
chains2.plot_2d(axes)
axes.iloc[0, 0].set_ylabel('')
axes.iloc[0, 0].set_xlabel(r'$M$ [GeV]')

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

prior_mins = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 110, 0, 0.1, 0, 0.1]
prior_maxs = [10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 150, 800, 10, 800, 10]

def nre_prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(17)]).T

def simulation(theta):

    exp1s, exp2s = [], []
    for i in tqdm(range(len(theta))):
        bgparams1 = theta[i, :6]
        bgparams2 = theta[i, 6:12]
        sig1params = [theta[i, 13], theta[i, 12], theta[i, 14]]
        sig2params = [theta[i, 15], theta[i, 12], theta[i, 16]]
        exp1s.append(poisson.rvs(background_model(normx, bgparams1) +
                                    signal_model(x, sig1params), size=len(x)))
        exp2s.append(poisson.rvs(background_model(normx, bgparams2) +
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

me1 = chains1['\mu'].values
w1 = chains1.get_weights()
me2 = chains2['\mu'].values
w2 = chains2.get_weights()
mej = chainsj['\mu'].values
wj = chainsj.get_weights()

sigmaD, sigmaA = [], []
for i in range(5):
    
    if load_trained_nre:
        nrei = nre.load(base_dir + 'nre_run' + str(i) + '.pkl',
              None, None, nre_prior)
    else:
        lr = ExponentialDecay(1e-3, 1000, 0.9)
        #lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, 1000, warmup_target=1e-1, warmup_steps=1000)
        nrei = nre(lr=lr)
        nrei.build_model(200, [120, 120], 'sigmoid')
            
        nrei.data_train = data_train
        nrei.data_test = data_test
        nrei.labels_train = labels_train
        nrei.labels_test = labels_test
        nrei.simulation_func_A = None
        nrei.simulation_func_B = None
        nrei.shared_prior = nre_prior
        nrei.prior_function_A = None
        nrei.prior_function_B = None

        nrei.training(epochs=1000, batch_size=1000, patience=50)
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

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3))
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
    
    #axes[2].hist(me1, bins=25, density=True, alpha=0.5, label='Exp1')
    #axes[2].hist(me2, bins=25, density=True, alpha=0.5, label='Exp2')
    #axes[2].hist(mej, bins=25, density=True, alpha=0.5, label='Joint')
    kde_plot_1d(axes[2], me1, label='Exp1', weights=w1)
    kde_plot_1d(axes[2], me2, label='Exp2', weights=w2)
    kde_plot_1d(axes[2], mej, label='Joint', weights=wj)
    axes[2].set_xlabel(r'Mass [GeV]')
    axes[2].set_ylabel('Density')
    axes[2].legend(loc='upper right')
    axes[2].set_yticks([])

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

sets_values = [[sigmaAs, sigmaA_lower, sigmaA_upper], [sigmaDs, sigmaD_lower, sigmaD_upper]]
for i in range(len(sets_values)):
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))
    axes.errorbar(np.arange(5), sets_values[i][0], yerr=[sets_values[i][1], sets_values[i][2]], fmt='o')
    axes.set_xticks(np.arange(5))
    if i == 0:
        axes.set_ylabel(r'$C$')
    else:
        axes.set_ylabel(r'$T$')
    axes.set_xlabel('Run')

    lower = 1/np.sqrt(5) * np.sqrt(np.sum((sets_values[i][1])**2))
    upper = 1/np.sqrt(5) * np.sqrt(np.sum((sets_values[i][2])**2))
    if i == 0:
        mean_values = mean_sigmaA
        print(mean_values, lower, upper)
    else:
        mean_values = mean_sigmaD
        print(mean_values, lower, upper)


    #for i in range(2):
    axes.axhline(mean_values, ls='--', c='r')
    axes.axhspan(mean_values - lower, mean_values + upper, alpha=0.1, color='r')
    if i ==0:
        axes.set_title(r'$C=$ ' + str(round(mean_values, 2)) + r'$^{+' + str(round(upper, 2)) + '}_{-' + str(round(lower, 2)) + '}$')
        plt.tight_layout()
        plt.savefig(base_dir + 'sigmaC.pdf', bbox_inches='tight')
    else:
        axes.set_title(r'$T=$ ' + str(round(mean_values, 2)) + r'$^{+' + str(round(upper, 2)) + '}_{-' + str(round(lower, 2)) + '}$')
        plt.tight_layout()
        plt.savefig(base_dir + 'sigmaT.pdf', bbox_inches='tight')
    plt.close()