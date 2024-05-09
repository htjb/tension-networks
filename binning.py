import matplotlib.pyplot as plt
import numpy as np
from tensionnet.utils import rebin

wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt('cosmology-data/wmap_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0, 0].plot(lwmap_raw, wmap_unbinned, label='Wmap Unbinned')
wmap_binned_like_wmap = rebin(wmap_unbinned, bins)

axes[0, 0].plot(wmap_data[:, 0], wmap_data[:, 3], marker='.', label='Wmap Released Binned')
axes[0, 0].plot(wmap_data[:, 0], wmap_binned_like_wmap, marker='.', label='Wmap Binned Mean')
axes[0, 0].legend()

axes[0, 1].plot(wmap_data[:, 0], np.abs(wmap_data[:, 3] - wmap_binned_like_wmap), marker='.')
axes[0, 1].set_title('ABS(Released Binned - My Binned)')
axes[0, 1].set_xlabel('Multipole')
axes[0, 1].set_ylabel('Difference')
axes[0, 0].set_xlabel('Multipole')
axes[0, 0].set_ylabel('Power Spectrum')

lplanck, signal_planck, _, _ = np.loadtxt('cosmology-data/planck_unbinned.txt', unpack=True)

axes[1, 0].plot(lplanck, signal_planck, label='Planck Unbinned')

# from the 2018 likelihood paper for planck section 3.2.5
bin1 = np.array([np.arange(2, 30, 1), np.arange(2, 30, 1)]).T
bin2 = np.array([np.arange(30, 100, 5), np.arange(30, 100, 5)+4]).T
bin3 = np.array([np.arange(100, 1503, 9), np.arange(100, 1503, 9)+8]).T
bin4 = np.array([np.arange(1504, 2013, 17), np.arange(1504, 2013, 17)+16]).T
bin5 = np.array([np.arange(2014, 2508, 33), np.arange(2014, 2508, 33)+32]).T
bins = np.vstack([bin1, bin2, bin3, bin4, bin5])
weights = lplanck*(lplanck+1)

indices = bins - 2
for i in range(len(bins)):
    print(i, 'cry')
plt.show()