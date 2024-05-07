import matplotlib.pyplot as plt
import numpy as np
from tensionnet.wmapplanck import  bin_planck
from cmblike.data import get_data

pwmap, lwmap = get_data(base_dir='cosmology-data/').get_wmap()

wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T

planck_binned_like_wmap = bin_planck(bins, lwmap)
np.savetxt('cosmology-data/planck_binned_like_wmap.txt', planck_binned_like_wmap)

plt.plot(lwmap, planck_binned_like_wmap*lwmap*(lwmap+1)/(2*np.pi))
plt.show()