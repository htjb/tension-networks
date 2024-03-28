import numpy as np
from cmblike.data import get_data
from tensionnet import wmapplanck
import matplotlib.pyplot as plt

wmapraw, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
praw, l = get_data(base_dir='cosmology-data/').get_planck()

nSamples = 50000
joint = wmapplanck.jointClGenCP(path='/Users/harrybevins/Documents/Software/cosmopower')

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

BASE_DIR = 'cosmopower-stuff/'
LOAD = False
label = '_diff_samples'
samples = np.array([[0.0107919,  0.12255735, 0.98870962, 2.94982524, 0.52871926]])

def prior(N):
    return np.array([np.random.uniform(prior_mins[i], prior_maxs[i], N) 
                     for i in range(len(parameters))]).T


if LOAD:
    samples = np.load(BASE_DIR + 'random_sample' + label + '.npy')
    planckExamples = np.load(BASE_DIR + 'random_planck_like_data' + label + '.npy')
    wmapExamples = np.load(BASE_DIR + 'random_wmap_like_data' + label + '.npy')
else:
    if samples is None:
        samples = prior(1)
    planckExamples, wmapExamples = joint(samples)
    np.save(BASE_DIR + 'random_planck_like_data' + label + '.npy', planckExamples)
    np.save(BASE_DIR + 'random_wmap_like_data' + label + '.npy', wmapExamples)
    np.save(BASE_DIR + 'random_sample' + label + '.npy', samples)

plt.plot(l, planckExamples[0]*l*(l+1)/(2*np.pi), label='Planck-Pretend')
plt.plot(lwmap, wmapExamples[0]*lwmap*(lwmap+1)/(2*np.pi), label='WMAP-Pretend')
plt.plot(l, praw*l*(l+1)/(2*np.pi), label='Planck')
plt.plot(lwmap, wmapraw*lwmap*(lwmap+1)/(2*np.pi), label='WMAP')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell$')
plt.savefig('pretend_wmap_planck_data_for_testing' + label + '.png', dpi=300, bbox_inches='tight')
plt.show()