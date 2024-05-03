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
data_label = ''

file = 'fit_wmap_binned_planck_cp/'
_, lobs = get_data(base_dir='cosmology-data/').get_wmap()
p = np.loadtxt('cosmology-data/planck_binned_like_wmap.txt')
pnoise = planck_noise(lobs).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/home/htjb2/rds/hpc-work/cosmopower')

likelihood = cmbs.get_likelihood(p, lobs, noise=pnoise, cp=True)
prior = cmbs.prior

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = BASE_DIR + file

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from fgivenx import plot_contours, plot_lines
import matplotlib.pyplot as plt
from anesthetic import read_chains

def signal():
    def signal_func(_, parameters):
        cl, sample = cmbs.get_samples(lobs, parameters, noise=pnoise)
        return sample*(lobs*(lobs+1))/(2*np.pi)
    return signal_func

fig, axes = plt.subplots(1)

samples = read_chains(BASE_DIR + file + 'test')
samples = samples.compress(1000)

names = ['p' + str(i) for i in range(5)]
samples = samples[names].values
sf = signal()
plot_lines(sf, lobs, samples, axes, color='r')
plt.plot(lobs, p*(lobs*(lobs+1))/(2*np.pi), c='k', label='Planck')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR + file + file.split('/')[0] + '.png', dpi=300)
plt.show()
