from pypolychord.settings import PolyChordSettings
import pypolychord
from cmblike.data import get_data
from cmblike.noise import planck_noise
from cmblike.cmb import CMB
import numpy as np
import matplotlib.pyplot as plt 

def rebin(signal, bins):
    indices = bins - 2
    binned_signal = []
    for i in range(len(indices)):
        if indices[i, 0] == indices[i, 1]:
            binned_signal.append(signal[int(indices[i, 0])])
        else:
            binned_signal.append(
                np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
    return np.array(binned_signal)

nDims = 5
nDerived = 0

RESUME = False
BASE_DIR = 'chains/'
data_label = ''

file = 'fit_wmap_binned_planck_cp_wide_prior/'
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lplanck, signal_planck, _, _ = np.loadtxt(
    'cosmology-data/planck_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

mask = lwmap > 0
lwmap = lwmap[mask]
bins = bins[mask]

pnoise = planck_noise(lwmap).calculate_noise()

planck_binned_like_wmap = rebin(signal_planck, bins)*2*np.pi/(lwmap*(lwmap+1))


parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
#prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
#prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]
prior_mins = [0.005, 0.001, 0.8, 1.61, 0.5]
prior_maxs = [0.1, 0.99, 1.2, 3.91, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

likelihood = cmbs.get_likelihood(planck_binned_like_wmap, lwmap, 
                                 noise=pnoise, cp=True, bins=bins)
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
        cl, sample = cmbs.get_samples(lwmap, parameters, noise=pnoise)
        return sample*(lwmap*(lwmap+1))/(2*np.pi)
    return signal_func

fig, axes = plt.subplots(1)

samples = read_chains(BASE_DIR + file + 'test')
samples = samples.compress(1000)

names = ['p' + str(i) for i in range(5)]
samples = samples[names].values
sf = signal()
plot_lines(sf, lwmap, samples, axes, color='r')
plt.plot(lwmap, planck_binned_like_wmap*(lwmap*(lwmap+1))/(2*np.pi), 
         c='k', label='Planck')
plt.xlabel(r'$l$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR + file + file.split('/')[0] + '.png', dpi=300)
plt.show()
