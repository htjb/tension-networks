import numpy as np
import matplotlib.pyplot as plt
from tensionnet.robs import run_poly
from tensionnet.bao import DESI_BAO, SDSS_BAO
from pypolychord.priors import UniformPrior, LogUniformPrior
from tensionnet.utils import plotting_preamble
from sklearn.model_selection import train_test_split
from anesthetic.plot import kde_contour_plot_2d
import camb
import tensorflow as tf
import random
from tqdm import tqdm
import os
from anesthetic import MCMCSamples

pars = camb.CAMBparams()

def derived(parameters):
    H0, rs, omm = [], [], []
    for i in tqdm(range(len(parameters))):
        pars.set_cosmology(H0=parameters[i][4]*100, ombh2=parameters[i][0], 
                                    omch2=parameters[i][1],
                                    tau=0.055)
        pars.InitPower.set_params(As=np.exp(parameters[i][3])/10**10, 
                            ns=parameters[i][2])
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(pars) # computes evolution of background cosmology

        H0.append(results.hubble_parameter(0))
        rs.append(results.get_derived_params()['rdrag']) # Mpc
        
        h = H0[-1]/100
        omb = parameters[i][0]/h**2
        omc = parameters[i][1]/h**2
        omm.append((omb+omc))

    H0 = np.array(H0)
    rs = np.array(rs)
    data = np.array(H0*rs)/100 # units of 100 Mpc
    #data /= 3e5
    data = np.vstack((data, omm, H0)).T

    samples = MCMCSamples(data=data, 
            labels=[r'$\frac{H_0 r_s}{c}$', r'$\Omega_m$', r'$H_0$'])
    return samples

plotting_preamble()

BASE_DIR = 'chains/full_desi_sdss_independent_observations/'
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

prior_mins = [0.005, 0.001, 0.8, 1.61, 0.5]
prior_maxs = [0.1, 0.99, 1.2, 3.91, 0.9]
desi_baos = DESI_BAO(data_location='cosmology-data/bao_data/', 
            prior_mins=prior_mins, prior_maxs=prior_maxs)
prior = desi_baos.prior
desi_likelihood = desi_baos.loglikelihood()

skip_polychord = True
if skip_polychord is False:
    file = BASE_DIR + 'DESI/'
    RESUME = True
    if RESUME is False:
        import os, shutil
        if os.path.exists(file):
            shutil.rmtree(file)

    run_poly(prior, desi_likelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)

    sdss_baos = SDSS_BAO(data_location='cosmology-data/bao_data/', 
                prior_mins=prior_mins, prior_maxs=prior_maxs)
    sdss_likelihood = sdss_baos.loglikelihood()

    file = BASE_DIR + 'SDSS/'
    RESUME = True
    if RESUME is False:
        import os, shutil
        if os.path.exists(file):
            shutil.rmtree(file)

    run_poly(prior, sdss_likelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)


    def joint_loglikelihood(theta):
        return desi_likelihood(theta)[0] + sdss_likelihood(theta)[0], []


    file = BASE_DIR + 'DESI_SDSS/'
    RESUME = True
    if RESUME is False:
        import os, shutil
        if os.path.exists(file):
            shutil.rmtree(file)

    run_poly(prior, joint_loglikelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)

##############################################################################
############################### Calculate R ##################################
##############################################################################

from anesthetic import read_chains

joint = read_chains(BASE_DIR + 'DESI_SDSS/test')
desi = read_chains(BASE_DIR + 'DESI/test')
sdss = read_chains(BASE_DIR + 'SDSS/test')

Rs = (joint.logZ(1000) - desi.logZ(1000) - sdss.logZ(1000))
R = Rs.mean()
errorR = Rs.std()
print('R:', R, '+/-', errorR)

joint = joint.compress()
desi = desi.compress()
sdss = sdss.compress()

##############################################################################
################################ Do NRE ######################################
##############################################################################

print('Running NRE...')
nSamples = 100000
load_data = True

def nre_prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(5)]).T

def simulation(theta):
    da, dh = [], []
    for i in tqdm(range(len(theta))):
        t = theta[i]
        try:
            sdss_sim = sdss_baos.get_sample(t)[0]
            desi_sim = desi_baos.get_sample(t)[:3]
            da.append([sdss_sim[0], sdss_sim[2], 
                       desi_sim[0][0], desi_sim[1][0], desi_sim[2][0]])
            dh.append([sdss_sim[1], sdss_sim[3], 
                       desi_sim[0][1], desi_sim[1][1], desi_sim[2][1]])
        except:
            pass
    da = np.array(da)
    dh = np.array(dh)

    idx = np.arange(len(da))
    np.random.shuffle(idx)
    
    shuffled_da = np.hstack([da[:, :2], da[idx, 2:]])
    shuffled_dh = np.hstack([dh[:, :2], dh[idx, 2:]])
    
    data = np.hstack([da, dh, np.array([[1]*len(da)]).T])
    idx = random.sample(range(len(data)), int(0.1*len(data)))

    data_validation = data[idx, :-1]
    data = np.delete(data, idx, axis=0)
    shuffled_da = np.delete(shuffled_da, idx, axis=0)
    shuffled_dh = np.delete(shuffled_dh, idx, axis=0)

    data_shuffled = np.hstack([shuffled_da, shuffled_dh, 
                               np.array([[0]*len(shuffled_da)]).T])
    data =  np.concatenate([data, data_shuffled])

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

    data_trainA = data_train[:, :len(da[0])]
    data_trainB = data_train[:, len(da[0]):]
    data_testA = data_test[:, :len(da[0])]
    data_testB = data_test[:, len(da[0]):]
    data_validationA = data_validation[:, :len(da[0])]
    data_validationB = data_validation[:, len(da[0]):]

    data_testA = (data_testA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_testB = (data_testB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)
    data_validationA = (data_validationA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_validationB = (data_validationB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)
    data_trainA = (data_trainA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_trainB = (data_trainB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)

    data_train = np.hstack([data_trainA, data_trainB])
    data_test = np.hstack([data_testA, data_testB])
    data_validation = np.hstack([data_validationA, data_validationB])

    return data_train, data_test, data_validation, labels_train, labels_test

from tensionnet.tensionnet import nre
from scipy.stats import ecdf
from tensionnet.utils import calcualte_stats
from tensorflow.keras.optimizers.schedules import ExponentialDecay  

load_trained_nre = True

if load_data:
    data_train = np.load(BASE_DIR + 'data_train.npy')
    data_test = np.load(BASE_DIR + 'data_test.npy')
    data_validation = np.load(BASE_DIR + 'data_validation.npy')
    labels_train = np.load(BASE_DIR + 'labels_train.npy')
    labels_test = np.load(BASE_DIR + 'labels_test.npy')
else:
    data_train, data_test, data_validation, labels_train, labels_test = simulation(nre_prior(nSamples))
    np.save(BASE_DIR + 'data_train.npy', data_train)
    np.save(BASE_DIR + 'data_test.npy', data_test)
    np.save(BASE_DIR + 'data_validation.npy', data_validation)
    np.save(BASE_DIR + 'labels_train.npy', labels_train)
    np.save(BASE_DIR + 'labels_test.npy', labels_test)

sigmaD, sigmaA = [], []
for i in range(5):
    
    if load_trained_nre:
        nrei = nre.load(BASE_DIR + 'nre_run' + str(i) + '.pkl',
              None, None, nre_prior)
    else:
        lr = ExponentialDecay(1e-3, 1000, 0.9)
        #lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, 1000, warmup_target=1e-1, warmup_steps=1000)
        nrei = nre(lr=lr)
        #nrei.build_model(6+4, [4]*2, 'sigmoid')
        nrei.build_compress_model(5, 5, [5, 5, 2], [4]*2,
                activation='sigmoid', compress='both', use_bias=True,)
            
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
        nrei.save(BASE_DIR + 'nre_run' + str(i) + '.pkl')

    plt.plot(nrei.loss_history, label='Training Loss')
    plt.plot(nrei.test_loss_history, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(BASE_DIR + 'loss_run' + str(i) + '.pdf', bbox_inches='tight')
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
    
    parameters = sdss.values[:, :5]
    bao_samples = derived(parameters).values
    kde_contour_plot_2d(ax=axes[2], data_x=bao_samples[:, 0], 
                        data_y=bao_samples[:, 1], alpha=0.5, label='SDSS')
    
    parameters = desi.values[:, :5]
    bao_samples = derived(parameters).values
    kde_contour_plot_2d(ax=axes[2], data_x=bao_samples[:, 0], 
                        data_y=bao_samples[:, 1], alpha=0.5, label='DESI')
    
    parameters = joint.values[:, :5]
    bao_samples = derived(parameters).values
    kde_contour_plot_2d(ax=axes[2], data_x=bao_samples[:, 0], 
                        data_y=bao_samples[:, 1], alpha=0.5, label='Joint')
    
    axes[2].set_xlabel(r'$H_0 r_s$ [100 Mpc]')
    axes[2].set_ylabel(r'$\Omega_m$')
    axes[2].legend()


    stats = calcualte_stats(R, errorR, c)
    print(stats)
    sigmaD.append(stats[:3])
    sigmaA.append(stats[3:6])

    plt.tight_layout()
    plt.savefig(BASE_DIR + 'desi_sdss_run' + str(i) + '.pdf', bbox_inches='tight')
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
"""axes[0].errorbar(np.arange(5), norm_sigmaA, yerr=[sigmaA_lower/mean_sigmaA[0], sigmaA_upper/mean_sigmaA[0]], fmt='o')
axes[0].set_xticks(np.arange(5))
axes[0].set_ylabel(r'$\sigma_A / \bar{\sigma}_A$')
axes[0].set_xlabel('Run')"""
axes.errorbar(np.arange(5), sigmaDs, yerr=[sigmaD_lower, sigmaD_upper], fmt='o')
axes.set_xticks(np.arange(5))
axes.set_ylabel(r'$T$')
axes.set_xlabel('Run')

lower_mean_sigmaD_error = 1/np.sqrt(5) * np.sqrt(np.sum((sigmaD_lower)**2))
upper_mean_sigmaD_error = 1/np.sqrt(5) * np.sqrt(np.sum((sigmaD_upper)**2))
print(mean_sigmaD, lower_mean_sigmaD_error, upper_mean_sigmaD_error)



#for i in range(2):
axes.axhline(mean_sigmaD, ls='--', c='r')
axes.axhspan(mean_sigmaD - lower_mean_sigmaD_error, mean_sigmaD + upper_mean_sigmaD_error, alpha=0.1, color='r')
plt.tight_layout()
plt.savefig(BASE_DIR + 'desi_sdss_sigma.pdf', bbox_inches='tight')
plt.close()