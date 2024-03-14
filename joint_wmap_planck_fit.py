import numpy as np
import matplotlib.pyplot as plt
from pypolychord.priors import UniformPrior
import camb
import healpy as hp
from cmblike.data import get_data
from tensionnet.robs import run_poly
from tqdm import tqdm
from random import shuffle
import matplotlib as mpl
from matplotlib import rc
import tensorflow as tf
from scipy.stats import ecdf
import pickle
import torch

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a',
     'f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

def prior(cube):
    # wide prior apart from tau which I left tight
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
    theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
    theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
    return theta

DATA_NORM = 'independent'
NUM_NETS = 12
HIDDEN_LAYERS = 1
REPEAT = 1

data = np.concatenate([praw, wmapraw]).astype(np.float32)
if REPEAT > 1:
    like = pickle.load(open('planck_wmap_likelihood_with_' + DATA_NORM + '_' + 
                        'data_norm_plus_batch_norm_' + str(NUM_NETS) + 
                        '_nets_number' + str(REPEAT) + '_' + str(HIDDEN_LAYERS) + '_hls.pkl', 'rb'))
else:
    like = pickle.load(open('planck_wmap_likelihood_with_' + DATA_NORM + '_' + 
                        'data_norm_plus_batch_norm_' + str(NUM_NETS) + 
                        '_nets_' + str(HIDDEN_LAYERS) + '_hls.pkl', 'rb'))

training_data = np.loadtxt('planck-wmap-nle-examples.txt').astype(np.float32)
training_params = np.loadtxt('planck-wmap-nle-params.txt').astype(np.float32)
training_planck = training_data[:, :len(praw)]
training_wmap = training_data[:, len(praw):]


if DATA_NORM == 'structured' or DATA_NORM == 'custom':
    std_planck = np.std(training_planck)
    std_wmap = np.std(training_wmap)
    stds = np.hstack([np.ones(len(praw))*1/std_planck, np.ones(len(wmapraw))*1/std_wmap])
    #correction = np.log(np.linalg.det(np.abs(np.diag(stds))))
    correction = np.linalg.slogdet(np.diag(stds))[1]

    # redefine data
    norm_praw = (praw - np.mean(training_planck)) / std_planck
    norm_wmapraw = (wmapraw - np.mean(training_wmap)) / std_wmap
    data = np.concatenate([norm_praw, norm_wmapraw]).astype(np.float32)

elif DATA_NORM == 'independent':
    std_planck = np.std(training_planck, axis=0)
    std_wmap = np.std(training_wmap, axis=0)
    stds = np.hstack([1/std_planck, 1/std_wmap])
    correction = np.linalg.slogdet(np.diag(stds))[1]

    # redefine data
    norm_praw = (praw - np.mean(training_planck, axis=0)) / std_planck
    norm_wmapraw = (wmapraw - np.mean(training_wmap, axis=0)) / std_wmap
    data = np.concatenate([norm_praw, norm_wmapraw]).astype(np.float32)
elif DATA_NORM == 'covariance':
    covariance = np.cov(training_data.T)
    invL = np.linalg.inv(np.linalg.cholesky(covariance))
    correction = np.linalg.slogdet(invL)[1]
    data = np.concatenate([praw, wmapraw])
    data = np.dot((data - np.mean(training_data, axis=0)), invL).astype(np.float32)



def likelihood(parameters):
    return (like.log_prob(torch.tensor([data]), 
                         torch.tensor([parameters.astype(np.float32)]),
                         ).detach().numpy() \
                                + correction)[0].astype(np.float64), []

"""print(correction)
params = np.array([0.022, 0.12, 1.04, 0.06, 0.96, 3.0])
print(likelihood(params))
sys.exit(1)
"""
nDerived = 0
nDims=6
RESUME=False

import pypolychord
from pypolychord.settings import  PolyChordSettings

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
if REPEAT > 1:
    settings.base_dir =  'wmap_planck_fit_with_' + DATA_NORM +'_data_norm' + \
        '_plus_batch_norm_' + str(NUM_NETS) + '_nets_number' + str(REPEAT) + '_' + str(HIDDEN_LAYERS) + '_hls/'
else:
    settings.base_dir =  'wmap_planck_fit_with_' + DATA_NORM +'_data_norm' + \
        '_plus_batch_norm_' + str(NUM_NETS) + '_nets_' + str(HIDDEN_LAYERS) + '_hls/'
#settings.nlive = 200*6 

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

chains = read_chains(settings.base_dir + '/test')
paramnames = ['p%i' for i in range(6)]
axes = chains.plot_2d(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'])
plt.savefig(settings.base_dir + '/2d.png')
plt.show()

planck = read_chains('Planck_fit/test')
wmap = read_chains('wmap_fit/test')

print((chains.logZ(1000) - planck.logZ(1000) - wmap.logZ(1000)).mean())