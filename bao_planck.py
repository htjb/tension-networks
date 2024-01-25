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

cmbs = CMB()
planck_likelihood = cmbs.get_likelihood(p, l, noise=planck_noise)
prior = cmbs.prior

baos = BAO(data_location='cosmology-data/bao_data/')
bao_likelihood = baos.loglikelihood()

def likelihood(theta):
    return planck_likelihood(theta) + bao_likelihood(theta), []

pars = camb.CAMBparams()

file = 'Planck_bao_fit/'
RESUME = False
nDims=6

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = file + '/'

output = pypolychord.run_polychord(likelihood, nDims, 0, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from anesthetic import read_chains

joint = read_chains('Planck_bao_fit/test')
planck = read_chains('Planck_fit/test')
bao = read_chains('bao_fit/test')

R = joint.logZ(10000) - planck.logZ(10000) - bao.logZ(10000)
R = R.values
print(np.mean(R), np.std(R))
