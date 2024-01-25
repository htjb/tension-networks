from pypolychord.settings import PolyChordSettings
import pypolychord
from cmblike.data import get_data
from cmblike.noise import wmap_noise
from cmblike.cmb import CMB
import numpy as np

nDims = 6
nDerived = 0

p, l = get_data(base_dir='cosmology-data/').get_wmap()
wmap_noise = wmap_noise(l).calculate_noise()

cmbs = CMB()

likelihood = cmbs.get_likelihood(p, l, noise=wmap_noise)
prior = cmbs.prior

file = 'wmap_fit/'
RESUME = True

settings = PolyChordSettings(nDims, 0) #settings is an object
settings.read_resume = RESUME
settings.base_dir = file + '/'

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

from fgivenx import plot_contours, plot_lines
import matplotlib.pyplot as plt
from anesthetic import read_chains

def signal():
    def signal_func(_, parameters):
        cl, sample = cmbs.get_samples(l, parameters, noise=wmap_noise)
        return sample*(l*(l+1))/(2*np.pi)
    return signal_func

fig, axes = plt.subplots(1)

samples = read_chains('wmap_fit/test')
samples = samples.compress()

names = ['p' + str(i) for i in range(6)]
samples = samples[names].values
plot_lines(signal, l, samples, axes, color='r')
plt.plot(l, p*(l*(l+1))/(2*np.pi), c='k', label='wmap')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig('wmap_fit.png', dpi=300)
plt.show()