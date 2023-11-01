import numpy as np
import matplotlib.pyplot as plt
from cmbemu.eval import evaluate
from anesthetic import read_chains

def load_planck():

    """
    Function to load in the planck power spectrum data.

    Returns
    -------
    p: power spectrum
    ps: the error on the power spectrum
    l_real: the multipoles
    """

    tt = np.loadtxt('TT_power_spec.txt', delimiter=',', dtype=str)

    l_real, p, ps, ns = [], [], [], []
    for i in range(len(tt)):
        if tt[i][0] == 'Planck binned      ':
            l_real.append(tt[i][2].astype('float')) # ell
            p.append(tt[i][4].astype('float')) # power spectrum
            ps.append(tt[i][6].astype('float')) # positive error
            ns.append(tt[i][5].astype('float')) # negative error
    p, ps, l_real = np.array(p), np.array(ps), np.array(l_real)
    return p, ps, l_real

p, _, l_real = load_planck()

samples = read_chains('Planck_chains/test')

predictor = evaluate(base_dir='cmbemu_model/', l=l_real)

def signal(l, theta):
    cl, _ = predictor(theta)
    return cl

from fgivenx import plot_contours, plot_lines
fig, axes = plt.subplots(1)
#samples = samples.compress()
print(samples)
names = ['p'+str(i) for i in range(6)]
samples = samples[names].values
#cbar = plot_contours(bao, z, samples, axes)
plot_lines(signal, l_real, samples, axes, color='r')
plt.plot(l_real, p, c='k')
plt.show()