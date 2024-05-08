import matplotlib.pyplot as plt
import numpy as np
from tensionnet.wmapplanck import jointClGenCP
from cmblike.cmb import CMB
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
    return np.array(binned_signal)*2*np.pi/(lwmap*(lwmap+1))

generator = jointClGenCP('/Users/harrybevins/Documents/Software/cosmopower')
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt('cosmology-data/wmap_unbinned.txt', unpack=True)
lplanck, signal_planck, _, _ = np.loadtxt('cosmology-data/planck_unbinned.txt', unpack=True)


bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

pnoise = planck_noise(lwmap).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

planck_binned_like_wmap = binning(signal_planck, bins)
wmap_binned_like_wmap = binning(wmap_unbinned, bins)

#np.savetxt('cosmology-data/planck_binned_like_wmap.txt', planck_binned_like_wmap)

#samples = prior(np.random.uniform(0, 1, 5))
samples = [0.022, 0.12, 0.96, 3.0448, 0.674]
pobs, wobs, cltheory = generator(samples, lwmap, bins)

from tensionnet.wmapplanck import loglikelihood

cltheory = cltheory[0]
pobs = pobs[0]
wobs = wobs[0]

A = lwmap*(lwmap+1)/(2*np.pi)

plt.plot(lwmap, cltheory*A, label='Theory')
plt.plot(lwmap, pobs*A, label='Planck')
plt.plot(lwmap, wobs*A, label='Wmap')
plt.plot(lwmap, planck_binned_like_wmap*A, label='Planck Binned')
plt.plot(lwmap, wmap_binned_like_wmap*A, label='Wmap Binned')
plt.legend()
plt.show()

like1= loglikelihood(planck_binned_like_wmap + pnoise, wmap_binned_like_wmap + wnoise, 
                    cltheory, pnoise, wnoise, lwmap)
like2 = loglikelihood(pobs, wobs, cltheory, pnoise, wnoise, lwmap)
print(like1, like2)