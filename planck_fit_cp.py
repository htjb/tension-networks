from pypolychord.settings import PolyChordSettings
import pypolychord
from cmblike.data import get_data
from cmblike.noise import planck_noise
from cmblike.cmb import CMB
import numpy as np

nDims = 5
nDerived = 0

RESUME = False
BASE_DIR = 'chains/'

file = 'planck_fit_cp_wide_prior/'
p, l = get_data(base_dir='cosmology-data/').get_planck()
planck_noise = planck_noise(l).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
#prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
#prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]
prior_mins = [0.005, 0.001, 0.8, 1.61, 0.5]
prior_maxs = [0.1, 0.99, 1.2, 3.91, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

likelihood = cmbs.get_likelihood(p, l, noise=planck_noise, cp=True)
prior = cmbs.prior

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)