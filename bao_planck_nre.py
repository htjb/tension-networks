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
from tqdm import tqdm

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
        pars.set_cosmology(H0=parameters[i][-1]*100, ombh2=parameters[i][0], 
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

p, l = get_data(base_dir='cosmology-data/').get_planck()
planck_noise = planck_noise(l).calculate_noise()

from tensionnet.bao import BAO
baos = BAO(data_location='cosmology-data/bao_data/')
z = baos.z

parameters = ['As', 'omegabh2', 'omegach2', 'ns', 'h']
prior_mins = [2.6, 0.01, 0.08, 0.8, 0.5]
prior_maxs = [3.8, 0.085, 0.21, 1.2, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs, 
           path_to_cp='/home/htjb2/rds/hpc-work/cosmopower')

def cl_func_gen():
    def cl_func(parameters):
        cl, sample = cmbs.get_samples(l, parameters, noise=planck_noise, cp=True)
        return sample
    return cl_func

def bao_func():
    def bao(parameters):
        datad12, datad16 = baos.get_camb_model(parameters)
        return np.concatenate((datad12, datad16))
    return bao

def signal_prior(n):
    theta = np.ones((n, 5))
    theta[:, 0] = np.random.uniform(0.01, 0.085, n) # omegabh2
    theta[:, 1] = np.random.uniform(0.08, 0.21, n) # omegach2
    #theta[:, 2] = np.random.uniform(0.97, 1.5, n) # 100*thetaMC
    #theta[:, 3] = np.random.uniform(0.01, 0.16, n) # tau
    theta[:, 2] = np.random.uniform(0.8, 1.2, n) # ns
    theta[:, 3] = np.random.uniform(2.6, 3.8, n) # log(10^10*As)
    theta[:, 4] = np.random.uniform(0.5, 0.9, n) # H0
    return theta

planck_func = cl_func_gen()
bao_func = bao_func()

from tensionnet.tensionnet import nre

nsamples = 250000
layers = [25]*5
from anesthetic import read_chains

joint = read_chains('chains/planck_bao_fit_cp/test')
planck = read_chains('chains/planck_fit_cp/test')
bao = read_chains('chains/bao_fit_h0/test')

R = joint.logZ(10000) - planck.logZ(10000) - bao.logZ(10000)
R = R.values
Rs, errorRs = np.mean(R), np.std(R)

try:
    nrei = nre.load('chains/planck_bao_fit_cp/bao_planck_model.pkl',
                planck_func, bao_func, signal_prior)
except:
    nrei = nre(lr=1e-4)
    nrei.build_model(len(l) + len(z)*2, 
                        layers, 'sigmoid')
    try:
        wide_data = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_nre_data.txt')
        wide_labels = np.loadtxt('chains/planck_bao_fit_cp/planck_bao_nre_labels.txt')
        nrei.data = wide_data
        nrei.labels = wide_labels
        nrei.simulation_func_A = planck_func
        nrei.simulation_func_B = bao_func
        nrei.shared_prior = signal_prior
    except:
        nrei.build_simulations(planck_func, bao_func, signal_prior, n=nsamples)
        np.savetxt('chains/planck_bao_fit_cp/planck_bao_data.txt', nrei.data)
        np.savetxt('chains/planck_bao_fit_cp/planck_bao_labels.txt', nrei.labels)
    model, data_test, labels_test = nrei.training(epochs=1000, batch_size=1000)
    nrei.save('chains/planck_bao_fit_cp/bao_planck_model.pkl')

nrei.__call__(iters=5000)
r = nrei.r_values
mask = np.isfinite(r)

fig, axes = plt.subplots(1, 3, figsize=(6.3, 6.3))
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
kde_contour_plot_2d(axes[2], bao_samples[:, 1], bao_samples[:, 2], alpha=0.5)

planck = planck.compress(1000)
parameters = planck.values[:, :5]
planck_samples = derived(parameters).values
kde_contour_plot_2d(axes[2], planck_samples[:, 1], planck_samples[:, 2], alpha=0.5)

joint = joint.compress(1000)
parameters = joint.values[:, :5]
joint_samples = derived(parameters).values
kde_contour_plot_2d(axes[2], joint_samples[:, 1], joint_samples[:, 2], alpha=0.5)

plt.tight_layout()
plt.savefig('figures/figure9.pdf', bbox_inches='tight')
plt.show()

