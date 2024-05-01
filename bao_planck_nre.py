import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import camb
import matplotlib as mpl
from matplotlib import rc
import scipy
from anesthetic import MCMCSamples
#from scipy.stats import ecdf
from cmblike.data import get_data
from cmblike.noise import planck_noise
from cmblike.cmb import CMB
from tensionnet.utils import calcualte_stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

pars = camb.CAMBparams()

def derived(parameters):
    H0, rs, omm = [], [], []
    for i in tqdm(range(len(parameters))):
        pars.set_cosmology(H0=parameters[i][4]*100, ombh2=parameters[i][0], 
                                    omch2=parameters[i][1],
                                    tau=0.055,
                                    theta_H0_range=[5, 1000])
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
    data = np.array(H0*rs)
    data /= 3e5
    data = np.vstack((data, omm, H0)).T

    samples = MCMCSamples(data=data, 
            labels=[r'$\frac{H_0 r_s}{c}$', r'$\Omega_m$', r'$H_0$'])
    return samples

_, l = get_data(base_dir='cosmology-data/').get_planck()
planck_noise = planck_noise(l).calculate_noise()

from tensionnet.bao import BAO
baos = BAO(data_location='cosmology-data/bao_data/')
z = baos.z

parameters = ['As', 'omegabh2', 'omegach2', 'ns', 'h']
prior_mins = [2.6, 0.01, 0.08, 0.8, 0.5]
prior_maxs = [3.8, 0.085, 0.21, 1.2, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs, 
           path_to_cp='/Users/harry/Documents/Software/cosmopower')

def cl_func_gen():
    def cl_func(parameters):
        cl, sample = cmbs.get_samples(l, parameters, noise=planck_noise, cp=True)
        return sample
    return cl_func

def bao_func():
    def bao(parameters):
        datad12, datad16, _, _ = baos.get_sample(parameters)
        return np.concatenate((datad12, datad16))
    return bao

def signal_prior(n):
    theta = np.ones((n, 5))
    theta[:, 0] = np.random.uniform(0.01, 0.085, n) # omegabh2
    theta[:, 1] = np.random.uniform(0.08, 0.21, n) # omegach2
    theta[:, 2] = np.random.uniform(0.8, 1.2, n) # ns
    theta[:, 3] = np.random.uniform(2.6, 3.8, n) # log(10^10*As)
    theta[:, 4] = np.random.uniform(0.5, 0.9, n) # H0
    return theta

planck_sim_func = cl_func_gen()
bao_sim_func = bao_func()

from tensionnet.tensionnet import nre

nsamples = 500000
from anesthetic import read_chains

joint = read_chains('chains/planck_bao_fit_cp/test')
planck = read_chains('chains/planck_fit_cp/test')
bao = read_chains('chains/bao_fit_h0/test')

R = joint.logZ(10000) - planck.logZ(10000) - bao.logZ(10000)
R = R.values
Rs, errorRs = np.mean(R), np.std(R)

PCA_planck = False
minmax = True
RETRAIN = True
if RETRAIN:
    import os
    if os.path.exists('chains/planck_bao_fit_cp/bao_planck_model.pkl'):
        os.remove('chains/planck_bao_fit_cp/bao_planck_model.pkl')

try:
    nrei = nre.load('chains/planck_bao_fit_cp/bao_planck_model.pkl',
                planck_sim_func, bao_sim_func, signal_prior)
except:
    from tensorflow.keras.optimizers.schedules import ExponentialDecay  
    lr = ExponentialDecay(1e-3, 1000, 0.9)
    #lr = 1e-3
    nrei = nre(lr=lr)
    #nrei.build_model(len(l) + len(z)*2, 
    #                    [100]*5, 'leaky_relu',
    #                    kernel_regularizer='l2')#, skip_layers=False)
                        #, kernel_regularizer=None)
    nrei.build_compress_model(len(l), len(z)*2,
                              [100, 100, 100, 100, 100, 50, 
                               50, 25, 25, 6, 6], [100]*10, 'swish',
                              compress='A')
    try:
        wide_data = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_data_250000.txt')
        wide_labels = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_labels_250000.txt')
        wide_data2 = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_data_100000.txt')
        wide_labels2 = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_labels_100000.txt')
        wide_data3 = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_data_150000.txt')
        wide_labels3 = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_labels_150000.txt')
        wide_data4 = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_data_500000.txt')
        wide_labels4 = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_labels_500000.txt')

        def dl_to_cl(sig, l):
            return sig * l * (l + 1) / (2*np.pi)
        
        wide_data = np.hstack([dl_to_cl(wide_data[:, :len(l)], l), wide_data[:, len(l):]])
        wide_data2 = np.hstack([dl_to_cl(wide_data2[:, :len(l)], l), wide_data2[:, len(l):]])
        wide_data3 = np.hstack([dl_to_cl(wide_data3[:, :len(l)], l), wide_data3[:, len(l):]])
        wide_data4 = np.hstack([dl_to_cl(wide_data4[:, :len(l)], l), wide_data4[:, len(l):]])
        nrei.data = np.vstack([wide_data, wide_data2, wide_data3, wide_data4])
        nrei.labels = np.hstack([wide_labels, wide_labels2, wide_labels3, wide_labels4])
        print(nrei.data.shape, nrei.labels.shape)

        data_train, data_test, labels_train, labels_test = \
                train_test_split(nrei.data, nrei.labels, test_size=0.3)
        
        data_trainA = data_train[:, :len(l)]
        data_trainB = data_train[:, len(l):]
        data_testA = data_test[:, :len(l)]
        data_testB = data_test[:, len(l):]

        idx = random.sample(range(len(data_trainA)), 500)
        testingA = data_trainA[idx]
        testingB = data_trainB[idx]

        data_trainA = np.delete(data_trainA, idx, axis=0)
        data_trainB = np.delete(data_trainB, idx, axis=0)
        labels_train = np.delete(labels_train, idx, axis=0)
        
        nrei.labels_test = labels_test
        nrei.labels_train = labels_train

        """L = np.linalg.inv(np.linalg.cholesky(np.cov(data_trainA, rowvar=False)))
        
        data_testA = L @ (data_testA - data_trainA.mean(axis=0)).T
        testingA = L @ (testingA - data_trainA.mean(axis=0)).T
        data_trainA = L @ (data_trainA - data_trainA.mean(axis=0)).T"""
        
        data_testB = np.hstack([data_testB[:, ::2], data_testB[:, 1::2]])
        testingB = np.hstack([testingB[:, ::2], testingB[:, 1::2]])
        data_trainB = np.hstack([data_trainB[:, ::2], data_trainB[:, 1::2]])
        
        # Use F_AP instead?
        """data_testB_DM = data_testB[:, ::2]
        data_testB_DH = data_testB[:, 1::2]
        testingB_DM = testingB[:, ::2]
        testingB_DH = testingB[:, 1::2]
        data_trainB_DM = data_trainB[:, ::2]
        data_trainB_DH = data_trainB[:, 1::2]

        data_testB, testingB, data_trainB = [], [], []
        for i in range(len(z)):
            data_testB.append((z[i]*data_testB_DM[:, i]**2*data_testB_DH[:, i]))
            testingB.append((z[i]*testingB_DM[:, i]**2*testingB_DH[:, i]))
            data_trainB.append((z[i]*data_trainB_DM[:, i]**2*data_trainB_DH[:, i]))
        data_testB = np.array(data_testB).T
        testingB = np.array(testingB).T
        data_trainB = np.array(data_trainB).T
        print(data_testB.shape)"""

        if minmax:
            data_testB = (data_testB - data_trainB.min(axis=0)) / \
                (data_trainB.max(axis=0) - data_trainB.min(axis=0))
            testingB = (testingB - data_trainB.min(axis=0)) / \
                (data_trainB.max(axis=0) - data_trainB.min(axis=0))
            data_trainB = (data_trainB - data_trainB.min(axis=0)) / \
                (data_trainB.max(axis=0) - data_trainB.min(axis=0))
        else:
            data_testB = (data_testB - data_trainB.mean(axis=0)) / \
                data_trainB.std(axis=0)
            testingB = (testingB - data_trainB.mean(axis=0)) / \
                data_trainB.std(axis=0)
            data_trainB = (data_trainB - data_trainB.mean(axis=0)) / \
                data_trainB.std(axis=0)
        
        if PCA_planck:
            cov = np.cov(data_trainA, rowvar=False)
            eig_vals, eig_vecs = np.linalg.eig(cov)

            args = np.argsort(eig_vals)[::-1]
            eig_vals = np.real(eig_vals[args])
            eig_vecs = eig_vecs[:, args]

            sum_eig_vals = np.sum(eig_vals)

            explained_vairance = eig_vals/sum_eig_vals
            cumulative_explained_variance = np.cumsum(explained_vairance)
            print(np.argmax(cumulative_explained_variance > 0.999) + 1)
            ncomponents = 24#np.argmax(cumulative_explained_variance > 0.99) + 1
            print(ncomponents)

            pca_data_trainA = np.dot(data_trainA, eig_vecs[:, :ncomponents])
            pca_data_testA = np.dot(data_testA, eig_vecs[:, :ncomponents])
            pca_testingA = np.dot(testingA, eig_vecs[:, :ncomponents])
            print(pca_data_trainA.shape)

            pca_data_testA = (pca_data_testA - pca_data_trainA.mean(axis=0)) / \
                pca_data_trainA.std(axis=0)
            pca_testingA = (pca_testingA - pca_data_trainA.mean(axis=0)) / \
                pca_data_trainA.std(axis=0)
            pca_data_trainA = (pca_data_trainA - pca_data_trainA.mean(axis=0)) / \
                pca_data_trainA.std(axis=0)

            nrei.data_train = np.hstack([pca_data_trainA, data_trainB])
            nrei.data_test = np.hstack([pca_data_testA, data_testB])
            testing_data = np.hstack([pca_testingA, testingB])
        else:
            if minmax:
                data_testA = (data_testA - data_trainA.min(axis=0)) / \
                    (data_trainA.max(axis=0) - data_trainA.min(axis=0))
                testingA = (testingA - data_trainA.min(axis=0)) / \
                    (data_trainA.max(axis=0) - data_trainA.min(axis=0))
                data_trainA = (data_trainA - data_trainA.min(axis=0)) / \
                    (data_trainA.max(axis=0) - data_trainA.min(axis=0))
            else:
                data_testA = (data_testA - data_trainA.mean(axis=0)) / \
                    data_trainA.std(axis=0)
                testingA = (testingA - data_trainA.mean(axis=0)) / \
                    data_trainA.std(axis=0)
                data_trainA = (data_trainA - data_trainA.mean(axis=0)) / \
                    data_trainA.std(axis=0)

            nrei.data_train = np.hstack([data_trainA, data_trainB])
            nrei.data_test = np.hstack([data_testA, data_testB])
            testing_data = np.hstack([testingA, testingB])

        nrei.simulation_func_A = planck_sim_func
        nrei.simulation_func_B = bao_sim_func
        nrei.prior_function_A = None
        nrei.prior_function_B = None
        nrei.shared_prior = signal_prior
    except:
        nrei.build_simulations(planck_sim_func, bao_sim_func, 
                               signal_prior, n=nsamples)
        np.savetxt('chains/planck_bao_fit_cp/planck_bao_data_500000.txt', nrei.data)
        np.savetxt('chains/planck_bao_fit_cp/planck_bao_labels_500000.txt', nrei.labels)
    model, data_test, labels_test = nrei.training(epochs=1000,
                                                  batch_size=1000)
    nrei.save('chains/planck_bao_fit_cp/bao_planck_model.pkl')


plt.figure(figsize=(5, 4))
plt.plot(nrei.loss_history, label='Training Loss')
plt.plot(nrei.test_loss_history, label='Testing Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('bao_planck_nre_loss.pdf')
plt.show()

nrei.__call__(iters=testing_data)
r = nrei.r_values
mask = np.isfinite(r)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
axes[0].hist(r[mask], bins=25,density=True)
axes[0].axvline(Rs, ls='--', c='r')
axes[0].set_title(r'$R_{obs}=$' + str(np.round(Rs, 2)) + r'$\pm$' +
                            str(np.round(errorRs, 2)))
axes[0].axvspan(Rs - errorRs, Rs + errorRs, alpha=0.1, color='r')
axes[0].set_xlabel(r'$\log R$')
axes[0].set_ylabel('Density')

rsort  = np.sort(r[mask])
c = scipy.stats.ecdf(rsort)

sigmaD, sigma_D_upper, sigma_D_lower, \
    sigmaA, sigma_A_upper, sigma_A_lower, \
        sigmaR, sigmaR_upper, sigmaR_lower = \
            calcualte_stats(Rs, errorRs, c)
print(f'Rs: {Rs}, Rs_upper: {Rs + errorRs},' + 
        f'Rs_lower: {Rs - errorRs}')
print(f'sigmaD: {sigmaD}, sigma_D_upper: ' + 
        f'{np.abs(sigmaD - sigma_D_upper)}, ' +
        f'sigma_D_lower: {np.abs(sigma_D_lower - sigmaD)}')
print(f'sigmaA: {sigmaA}, sigma_A_upper: ' +
        f'{np.abs(sigmaA - sigma_A_upper)}, ' +
        f'sigma_A_lower: {np.abs(sigma_A_lower - sigmaA)}')
print(f'sigmaR: {sigmaR}, sigmaR_upper: ' + 
        f'{np.abs(sigmaR - sigmaR_upper)}, ' +
        f'sigmaR_lower: {np.abs(sigmaR_lower - sigmaR)}')
np.savetxt('chains/planck_bao_fit_cp/tension_stats.txt',
            np.hstack([sigmaD, sigma_D_upper, sigma_D_lower,
                        sigmaA, sigma_A_upper, sigma_A_lower,
                        sigmaR, sigmaR_upper, sigmaR_lower]).T)

axes[1].plot(rsort, c.cdf.evaluate(rsort)) 
axes[1].axhline(c.cdf.evaluate(Rs), ls='--',
        color='r')
axes[1].axhspan(c.cdf.evaluate(Rs - errorRs), 
        c.cdf.evaluate(Rs + errorRs), 
        alpha=0.1, 
        color='r')
axes[1].set_xlabel(r'$\log R$')
axes[1].set_ylabel(r'$P(\log R < \log R^\prime)$')
axes[1].set_title(r'$\sigma_D =$' + f'{sigmaD:.3f}' + 
                         r'$+$' + f'{np.abs(sigmaD - sigma_D_upper):.3f}' +
                r'$(-$' + f'{np.abs(sigma_D_lower - sigmaD):.3f}' + r'$)$' + '\n' +
                r'$\sigma_A=$' + f'{sigmaA:.3f}' + 
                r'$+$' + f'{np.abs(sigmaA - sigma_A_upper):.3f}' +
                r'$(-$' + f'{np.abs(sigma_A_lower - sigmaA):.3f}' + r'$)$')

from anesthetic.plot import kde_contour_plot_2d

bao = bao.compress(1000)
parameters = bao.values[:, :5]
bao_samples = derived(parameters).values
kde_contour_plot_2d(axes[2], bao_samples[:, 1], bao_samples[:, 2], alpha=0.5, label='BAO')

planck = planck.compress(1000)
parameters = planck.values[:, :5]
planck_samples = derived(parameters).values
kde_contour_plot_2d(axes[2], planck_samples[:, 1], planck_samples[:, 2], alpha=0.5,
                    label='Planck')

joint = joint.compress(1000)
parameters = joint.values[:, :5]
joint_samples = derived(parameters).values
kde_contour_plot_2d(axes[2], joint_samples[:, 1], joint_samples[:, 2], alpha=0.5, label='Joint')
axes[2].legend()
axes[2].set_xlabel(r'$\Omega_m$')
axes[2].set_ylabel(r'$H_0$')

plt.tight_layout()
plt.savefig('figures/figure9.pdf', bbox_inches='tight')
plt.show()

