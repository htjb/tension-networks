import matplotlib.pyplot as plt
import numpy as np
from scipy.special import loggamma, hyp0f1
from numpy.linalg import slogdet
from tensionnet.wmapplanck import jointClGenCP, bin_planck
from cmblike.cmb import CMB
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise

porig, l = get_data(base_dir='cosmology-data/').get_planck()
pwmap, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
p = np.interp(lwmap, l, porig) 

pnoise = planck_noise(lwmap).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harry/Documents/Software/cosmopower')

prior = cmbs.prior

generator = jointClGenCP(cmbs.path_to_cp)
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T

planck_binned_like_wmap = bin_planck(porig, bins)
np.savetxt('cosmology-data/planck_binned_like_wmap.txt', planck_binned_like_wmap)

#samples = prior(np.random.uniform(0, 1, 5))
samples = [0.022, 0.12, 0.96, 3, 0.674]
pobs, wobs, crossobs, cltheory = generator(samples, lwmap, bins)
plt.plot(lwmap, pobs*lwmap*(lwmap+1)/(2*np.pi), marker='.', label='Planck-wmap bins')
plt.plot(lwmap, wobs*lwmap*(lwmap+1)/(2*np.pi), label='wmap')
plt.plot(lwmap, cltheory*lwmap*(lwmap+1)/(2*np.pi), label='Theory')
plt.plot(lwmap, pnoise*lwmap*(lwmap+1)/(2*np.pi), label='Planck noise')
plt.plot(lwmap, wnoise*lwmap*(lwmap+1)/(2*np.pi), label='wmap noise')
plt.ylim([0, 10000])
plt.legend()
plt.show()

from tensionnet.wmapplanck import loglikelihood

print(loglikelihood(pobs, wobs, cltheory, pnoise, wnoise, lwmap))

plt.plot(lwmap, planck_binned_like_wmap, label='Planck binned', marker='.')
plt.plot(lwmap, pwmap*lwmap*(lwmap+1)/(2*np.pi), label='wmap', marker='.')
plt.plot(l, porig*l*(l+1)/(2*np.pi), marker='.', label='Planck binned by planck')
plt.legend()
plt.show()