import matplotlib.pyplot as plt
import numpy as np
from scipy.special import loggamma, hyp0f1
from numpy.linalg import slogdet
from tensionnet.wmapplanck import jointClGenCP, bin_planck
from cmblike.cmb import CMB
from cmblike.data import get_data
from cmblike.noise import planck_noise, wmap_noise

def binning(signal, bins):
    indices = bins - 2
    binned_signal = []
    for i in range(len(indices)):
        if indices[i, 0] == indices[i, 1]:
            binned_signal.append(signal[int(indices[i, 0])])
        else:
            binned_signal.append(
                np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
    return np.array(binned_signal)#*2*np.pi/(lwmap*(lwmap+1))

wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt('cosmology-data/wmap_unbinned.txt', unpack=True)
lplanck, signal_planck, _, _ = np.loadtxt('cosmology-data/planck_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(lwmap_raw, wmap_unbinned, label='Wmap Unbinned')
wmap_binned_like_wmap = binning(wmap_unbinned, bins)

axes[0].plot(wmap_data[:, 0], wmap_data[:, 3], marker='.', label='Wmap Released Binned')
axes[0].plot(wmap_data[:, 0], wmap_binned_like_wmap, marker='.', label='Wmap Binned Mean')
axes[0].legend()

axes[1].plot(wmap_data[:, 0], np.abs(wmap_data[:, 3] - wmap_binned_like_wmap), marker='.')
axes[1].set_title('ABS(Released Binned - My Binned)')
axes[1].set_xlabel('Multipole')
axes[1].set_ylabel('Difference')
axes[0].set_xlabel('Multipole')
axes[0].set_ylabel('Power Spectrum')
plt.show()