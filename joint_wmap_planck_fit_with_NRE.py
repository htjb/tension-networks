import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
from anesthetic import read_chains
from cmblike.data import get_data
from cmblike.noise import wmap_noise
from cmblike.cmb import CMB
from tensionnet.robs import run_poly
from tqdm import tqdm
from random import shuffle
import matplotlib as mpl
from matplotlib import rc
import tensorflow as tf

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a',
     'f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

PRETEND_DATA = False
data_label = '_diff_samples'

if PRETEND_DATA:
    praw = np.load('cosmopower-stuff/random_planck_like_data' + data_label + '.npy')[0]
    wmapraw = np.load('cosmopower-stuff/random_wmap_like_data' + data_label + '.npy')[0]
    _, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
    _, l = get_data(base_dir='cosmology-data/').get_planck()
else:
    wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
    praw, l = get_data(base_dir='cosmology-data/').get_planck()

if PRETEND_DATA:
    planckchains = read_chains('cosmopower-stuff/pretend_planck_fit_with_cp_no_tau' + data_label + '/test')
    wmapchains = read_chains('cosmopower-stuff/pretend_wmap_fit_with_cp_no_tau' + data_label + '/test')
else:
    planckchains = read_chains('cosmopower-stuff/planck_fit_with_cp_no_tau/test')
    wmapchains = read_chains('cosmopower-stuff/wmap_fit_with_cp_no_tau/test')

planckEvidence = planckchains.logZ(1000).mean()

def prior(cube):
    # wide prior apart from tau which I left tight
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.005, 0.04)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
    theta[2] = UniformPrior(0.8, 1.2)(cube[2]) # ns
    theta[3] = UniformPrior(2.6, 3.8)(cube[3]) # log(10^10*As)
    theta[4] = UniformPrior(0.5, 0.9)(cube[4]) # h
    return theta

ratio_estimator = tf.keras.models.load_model('cosmopower-stuff/' +
                                             'cosmopower_joint_likelihood.keras')

if PRETEND_DATA:
    BASE_DIR = 'cosmopower-stuff/pretend_wmap_planck_nre_no_tau' + data_label + '/'
else:
    BASE_DIR = 'cosmopower-stuff/wmap_planck_nre_no_tau/'

train_planck_mean = np.loadtxt('cosmopower-stuff/train_planck_mean.txt')
train_planck_std = np.loadtxt('cosmopower-stuff/train_planck_std.txt')
train_wmap_mean = np.loadtxt('cosmopower-stuff/train_wmap_mean.txt')
train_wmap_std = np.loadtxt('cosmopower-stuff/train_wmap_std.txt')
train_params_mean = np.loadtxt('cosmopower-stuff/train_params_mean.txt')
train_params_std = np.loadtxt('cosmopower-stuff/train_params_std.txt')

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]
cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

wn = wmap_noise(lwmap).calculate_noise()
wmaplikelihood = cmbs.get_likelihood(wmapraw, lwmap, noise=wn, cp=True)

# redefine data
norm_praw = (praw - train_planck_mean) / train_planck_std
norm_wmapraw = (wmapraw - train_wmap_mean) / train_wmap_std

def likelihood(theta):

    wmaplike = wmaplikelihood(theta)[0]
    theta = (theta - train_params_mean) / train_params_std
    ps = tf.convert_to_tensor(np.array([[*norm_wmapraw, 
                *theta, *norm_praw]]).astype('float32'))
    logr = ratio_estimator(ps).numpy()[0]
    likelihood = logr + wmaplike + \
                    planckEvidence
    print(likelihood, logr, wmaplike, planckEvidence)

    return likelihood[0].astype(np.float64), []


params = np.array([0.022, 0.12, 0.96, 3.0, 0.674])
print(likelihood(params))
sys.exit(1)

nDerived = 0
nDims=5
RESUME=True

import pypolychord
from pypolychord.settings import  PolyChordSettings

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR
#settings.nlive =

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)


chains = read_chains(settings.base_dir + '/test')
paramnames = ['p%i' for i in range(nDims)]
axes = chains.plot_2d(['p0', 'p1', 'p2', 'p3', 'p4'], alpha=0.5, label='Joint',
                      figsize=(6.3, 6.3),
                      )

if PRETEND_DATA:
    true_params = np.load('cosmopower-stuff/random_sample' + data_label + '.npy')[0]
    axes.axlines(({f'p{i}': true_params[i] for i in range(5)}), color='k', ls='--')
    planck_chains = read_chains('cosmopower-stuff/pretend_planck_fit_with_cp_no_tau' + data_label + '/test')
    wmap_chains = read_chains('cosmopower-stuff/pretend_wmap_fit_with_cp_no_tau' + data_label + '/test')
    planck_chains.plot_2d(axes, alpha=0.5, label='Planck-Chains')
    wmap_chains.plot_2d(axes, alpha=0.5, label='WMAP-Chains')
else:
    planck_best_fit = [0.022, 0.12, 0.96, 3.04, 0.674]
    axes.axlines({f'p{i}': planck_best_fit[i] for i in range(5)}, color='k', ls='--', label='Planck')
    """
    the 9 yr wmap results give curavture purturbations as 10^9 \Delta_R^2 = 2.40
    I think this is As but As is more traditionally given as ln(10^10 As)
    where As=\Delta_R^2 so converting to the common units gives 3.18...
    hmm looks to big...
    """
    wmap_best_fit = [0.022, 0.1142, 0.97, np.nan, 0.69]
    axes.axlines({f'p{i}': wmap_best_fit[i] for i in range(5)}, color='r', ls='--', label='WMAP')
    planck_chains = read_chains('cosmopower-stuff/planck_fit_with_cp_no_tau/test')
    wmap_chains = read_chains('cosmopower-stuff/wmap_fit_with_cp_no_tau/test')
    planck_chains.plot_2d(axes, alpha=0.5, label='Planck-Chains')
    wmap_chains.plot_2d(axes, alpha=0.5, label='WMAP-Chains')

axes.iloc[0, 0].legend(bbox_to_anchor=(1, 1.2), loc='lower left', ncols=3)
axis_labels = [r'$\Omega_b h^2$', r'$\Omega_c h^2$', 
               r'$n_s$', r'$\log(10^{10} A_s)$', r'$h$']
for i in range(len(axes.iloc[0, :])):
    axes.iloc[-1, i].set_xlabel(axis_labels[i])
for i in range(len(axes.iloc[:, 0])):
    axes.iloc[i, 0].set_ylabel(axis_labels[i])
plt.tight_layout()
plt.savefig(settings.base_dir + '/2d.png', dpi=300, bbox_inches='tight')
plt.show()

print((chains.logZ(1000) - planckchains.logZ(1000) - wmapchains.logZ(1000)).mean())