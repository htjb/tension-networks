import matplotlib.pyplot as plt
import numpy as np
from scipy.special import loggamma, hyp0f1
from numpy.linalg import slogdet
from tensionnet.wmapplanck import jointClGenCP
from cmblike.cmb import CMB
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise

p, l = get_data(base_dir='cosmology-data/').get_planck()
pwmap, lwmap = get_data(base_dir='cosmology-data/').get_wmap()
p = np.interp(lwmap, l, p) 
pnoise = planck_noise(lwmap).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
prior_mins = [0.005, 0.08, 0.8, 2.6, 0.5]
prior_maxs = [0.04, 0.21, 1.2, 3.8, 0.9]

cmbs = CMB(parameters=parameters, prior_mins=prior_mins, 
           prior_maxs=prior_maxs,
           path_to_cp='/Users/harrybevins/Documents/Software/cosmopower')

prior = cmbs.prior

generator = jointClGenCP(cmbs.path_to_cp)

samples = np.array([prior(np.random.uniform(0, 1, 5)) for i in range(100)])
pobs, wobs, crossobs, cltheory = generator(samples, lwmap)
plt.plot(lwmap, pobs[0]*lwmap*(lwmap+1)/(2*np.pi))
plt.plot(lwmap, wobs[0]*lwmap*(lwmap+1)/(2*np.pi))
plt.plot(lwmap, crossobs[0]*lwmap*(lwmap+1)/(2*np.pi))
plt.plot(lwmap, cltheory[0]*lwmap*(lwmap+1)/(2*np.pi))
plt.ylim([0, 10000])
plt.show()

def logpdf(hatCF, hatCG, C, NF, NG, l):
    D = ((C+NF)*(C+NG) - C**2)/(2*l+1)
    logp = -2*loggamma((2*l+1)/2) - (2*l+1)/2*np.log(4*D/(2*l+1)) - \
        ((C+NG)*hatCF + (C+NF)*hatCG)/(2*D) + (2*l-1)/2*np.log(hatCF*hatCG) + \
            np.log(hyp0f1((2*l+1)/2, hatCF*hatCG*C**2/4/D**2))
    return np.where(np.isfinite(logp), logp, -np.inf)

print(logpdf(pobs[0], wobs[0], cltheory[0], pnoise, wnoise, lwmap))
plt.plot(lwmap, (pobs[0] - wobs[0])*lwmap*(lwmap+1)/(2*np.pi))
plt.show()


"""C = 1
NF = 0.1
NG = 1
l = 10

N = 10000
a = np.sqrt(C) * np.random.randn(2*l+1, N)
hataF = a + np.sqrt(NF)*np.random.randn(2*l+1, N)
hataG = a + np.sqrt(NG)*np.random.randn(2*l+1, N)
hatCF = (hataF**2).mean(axis=0)
hatCG = (hataG**2).mean(axis=0)
print(hatCF)

plt.tricontour(hatCF, hatCG, logpdf(hatCF, hatCG, C, NF, NG, l))
plt.plot(hatCF, hatCG, '.', alpha=0.1)
plt.show()

from scipy.integrate import dblquad
func = lambda hatCF, hatCG: np.exp(logpdf(hatCF, hatCG, C, NF, NG, l))
print(dblquad(func, 0, np.inf, 0, np.inf))"""
