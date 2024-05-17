import numpy as np
import matplotlib.pyplot as plt
from tensionnet.robs import run_poly
from tensionnet.bao import DESI_BAO, SDSS_BAO
from pypolychord.priors import UniformPrior, LogUniformPrior
import camb

prior_mins = [0.005, 0.001, 0.8, 1.61, 0.5]
prior_maxs = [0.1, 0.99, 1.2, 3.91, 0.9]
desi_baos = DESI_BAO(data_location='cosmology-data/bao_data/', 
            prior_mins=prior_mins, prior_maxs=prior_maxs)
prior = desi_baos.prior
desi_likelihood = desi_baos.loglikelihood()

file = 'All_the_BAOs/DESI/'
RESUME = True
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, desi_likelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)

sdss_baos = SDSS_BAO(data_location='cosmology-data/bao_data/', 
            prior_mins=prior_mins, prior_maxs=prior_maxs)
sdss_likelihood = sdss_baos.loglikelihood()

file = 'All_the_BAOs/SDSS/'
RESUME = True
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, sdss_likelihood, file, RESUME=RESUME, nDims=5, nlive=25*5)


def joint_loglikelihood(theta):
    return desi_likelihood(theta)[0] + sdss_likelihood(theta)[0], []


file = 'All_the_BAOs/DESI_SDSS/'
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

joint = read_chains('All_the_BAOs/DESI_SDSS/test')
desi = read_chains('All_the_BAOs/DESI/test')
sdss = read_chains('All_the_BAOs/SDSS/test')

R = (joint.logZ(1000) - desi.logZ(1000) - sdss.logZ(1000))
R = R.mean()
errorR = R.std()


##############################################################################
################################ Do NRE ######################################
##############################################################################

print('Running NRE...')
nSamples = 50000
load_data =True

def nre_prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(5)]).T

def sdss_simulation(theta):
    return np.concatenate(sdss_baos.get_sample(theta)[:2])

def desi_simulation(theta):
    return np.concatenate(desi_baos.get_sample(theta)[:2])

from tensionnet.tensionnet import nre
from scipy.stats import ecdf
from tensionnet.utils import calcualte_stats
from tensorflow.keras.optimizers.schedules import ExponentialDecay  

lr = ExponentialDecay(1e-3, 1000, 0.9)
nrei = nre(lr=lr)
nrei.build_model(6+4, [6]*5, 'sigmoid', 
                 skip_layers=False) 
#nrei.build_compress_model(6, 4, [6, 6, 6, 2, 2], [5]*5,
#        activation='sigmoid', compress='both',
#        compress_layer_sizesB=[4, 4, 2, 2], use_bias=True, kernel_regularizer=None,)

if load_data:
    data_train1 = np.loadtxt('All_the_BAOs/data_train.txt')
    data_test1 = np.loadtxt('All_the_BAOs/data_test.txt')
    label_train1 = np.loadtxt('All_the_BAOs/labels_train.txt')
    label_test1 = np.loadtxt('All_the_BAOs/labels_test.txt')
    data_train2 = np.loadtxt('All_the_BAOs/data_train2.txt')
    data_test2 = np.loadtxt('All_the_BAOs/data_test2.txt')
    label_train2 = np.loadtxt('All_the_BAOs/labels_train2.txt')
    label_test2 = np.loadtxt('All_the_BAOs/labels_test2.txt')

    """data_train1_da = data_train1[:, ::2]
    data_train1_dh = data_train1[:, 1::2]
    data_test1_da = data_test1[:, ::2]
    data_test1_dh = data_test1[:, 1::2]
    data_train2_da = data_train2[:, ::2]
    data_train2_dh = data_train2[:, 1::2]
    data_test2_da = data_test2[:, ::2]
    data_test2_dh = data_test2[:, 1::2]

    data_train1 = data_train1_da**2*data_train1_dh
    data_test1 = data_test1_da**2*data_test1_dh
    data_train2 = data_train2_da**2*data_train2_dh
    data_test2 = data_test2_da**2*data_test2_dh"""

    sdss_redshift = np.concatenate([sdss_baos.d12[:, 0], sdss_baos.d16[:, 0]])
    desi_redshift = np.concatenate([desi_baos.L1[:, 0], desi_baos.L2[:, 0]])
    #print(sdss_redshift, desi_redshift)
    redshifts = np.concatenate([sdss_redshift, desi_redshift])

    """for i in range(100):
        plt.plot(sdss_redshift, data_train1[i, :6], c='r', ls='', marker='o')
        plt.plot(desi_redshift, data_train1[i, 6:], c='b', ls='', marker='o')
    plt.show()
    exit()"""
    
    nrei.data_train = np.concatenate([data_train1, data_train2])
    nrei.data_test = np.concatenate([data_test1, data_test2])
    nrei.labels_train = np.concatenate([label_train1, label_train2])
    nrei.labels_test = np.concatenate([label_test1, label_test2])
    nrei.simulation_func_A = sdss_simulation
    nrei.simulation_func_B = desi_simulation
    nrei.shared_prior = nre_prior
    nrei.prior_function_A = None
    nrei.prior_function_B = None
else:
    nrei.build_simulations(sdss_simulation, desi_simulation, 
                        nre_prior, nSamples)
    np.savetxt('All_the_BAOs/data_test2.txt', nrei.data_test)
    np.savetxt('All_the_BAOs/data_train2.txt', nrei.data_train)
    np.savetxt('All_the_BAOs/labels_test2.txt', nrei.labels_test)
    np.savetxt('All_the_BAOs/labels_train2.txt', nrei.labels_train)

nrei.training(epochs=1000, batch_size=10000)

nrei.__call__(iters=1000)
r = nrei.r_values
mask = np.isfinite(r)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 4))
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

plt.tight_layout()
plt.savefig('All_the_BAOs/desi_sdss.pdf', bbox_inches='tight')
plt.close()

