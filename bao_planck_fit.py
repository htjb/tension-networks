import numpy as np
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings
import pypolychord
import camb
import matplotlib.pyplot as plt
from scipy.stats import chi2
from cmblike.data import get_data
from cmblike.noise import planck_noise
from cmblike.cmb import CMB
from tensionnet.bao import BAO


p, l = get_data(base_dir='cosmology-data/').get_planck()
pnoise = planck_noise(l).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
#prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
#prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]
prior_mins = [0.005, 0.001, 0.8, 1.61, 0.5]
prior_maxs = [0.1, 0.99, 1.2, 3.91, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/home/htjb2/rds/hpc-work/cosmopower')

planck_likelihood = cmbs.get_likelihood(p, l, noise=pnoise, cp=True)
prior = cmbs.prior

baos = BAO(data_location='cosmology-data/bao_data/')
bao_likelihood = baos.loglikelihood()

def likelihood(theta):
    return planck_likelihood(theta)[0] + bao_likelihood(theta)[0], []

pars = camb.CAMBparams()

file = 'chains/planck_bao_fit_cp_wide_prior/'
RESUME = True
nDims=5

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = file + '/'
settings.nlive = 25*5

output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

joint = read_chains('chains/planck_bao_fit_cp/test')
planck = read_chains('chains/planck_fit_cp/test')
bao = read_chains('chains/bao_fit_h0/test')

R = joint.logZ(10000) - planck.logZ(10000) - bao.logZ(10000)
R = R.values
print(np.mean(R), np.std(R))
