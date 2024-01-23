import numpy as np
import matplotlib.pyplot as plt
from tensionnet.robs import run_poly
from tensionnet.bao import BAO
from pypolychord.priors import UniformPrior, LogUniformPrior
import camb

baos = BAO(data_location='cosmology-data/bao_data/')
prior = baos.prior
likelihood = baos.loglikelihood()

file = 'bao_fit/'
RESUME = False
if RESUME is False:
    import os, shutil
    if os.path.exists(file):
        shutil.rmtree(file)

run_poly(prior, likelihood, file, RESUME=RESUME, nDims=6, nlive=200*6)


from anesthetic import read_chains

samples = read_chains('bao_fit/test')
names = ['p'+str(i) for i in range(6)]

z = baos.z

def bao_da(z, theta):
    data12, data16 = baos.get_camb_model(theta)
    return [data12[0], data12[2], data16[0]]

def bao_dh(z, theta):
    data12, data16 = baos.get_camb_model(theta)
    return [data12[1], data12[3], data16[1]]


from fgivenx import plot_contours, plot_lines
fig, axes = plt.subplots(1)
#samples = samples.compress()
print(samples)
#cbar = plot_contours(bao, z, samples, axes)
plot_lines(bao_da, z, samples, axes, color='r')
plot_lines(bao_dh, z, samples, axes, color='b')
plt.plot(baos.d12[::2,0], baos.d12[::2,1], '*', label='DR12 DM', c='k')
plt.plot(baos.d12[1::2,0], baos.d12[1::2,1], '^', label='DR12 DH', c='k')
plt.plot(baos.d16[::2,0], baos.d16[::2,1], 'o', label='DR16 DM', c='k')
plt.plot(baos.d16[1::2,0], baos.d16[1::2,1], '+', label='DR16 DH', c='k')
plt.xlabel(r'$z$')
plt.ylabel(r'$D/r_s$')
plt.legend()
plt.tight_layout()
plt.savefig('bao_fit.png', dpi=300)
plt.show()