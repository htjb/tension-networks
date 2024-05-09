import matplotlib.pyplot as plt
import numpy as np
from tensionnet.wmapplanck import jointClGenCP
from cmblike.cmb import CMB
from cmblike.noise import planck_noise, wmap_noise
from scipy.special import loggamma, hyp0f1
from numpy.linalg import slogdet
from scipy.special import ive, iv

def loghyp0f1(l, x):
    """log(hyp0f1((2*l+1)/2, x**2))"""
    ans1 = np.log(hyp0f1((2*l+1)/2, x**2))
    with np.errstate(divide='ignore'):
        ans2 = np.log(ive((2*l-1)/2,2*x)) + 2*x + loggamma((2*l+1)/2) -(2*l-1)/2*np.log(x)
    ans3 = 2*x - l*np.log(x) + loggamma((2*l+1)/2) - np.log(4*np.pi)/2 
    ans = ans1
    ans = np.where(np.isfinite(ans), ans, ans2)
    ans = np.where(np.isfinite(ans), ans, ans3)
    return ans, ans1, ans2, ans3

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

def loglikelihood(hatCF, hatCG, C, NF, NG, l):
    """ takes in the observed power spectra, theory, noise and relevant l"""

    D = ((C+NF)*(C+NG) - C**2)/(2*l+1)
    l1 = -2*loggamma((2*l+1)/2) 
    l2 = - (2*l+1)/2*np.log(4*D/(2*l+1)) 
    l3A = - ((C+NG)*hatCF)/(2*D)
    l3B = -((C+NF)*hatCG)/(2*D)
    l4 = (2*l-1)/2*np.log(hatCF*hatCG)
    logp = l1 + l2 + l3A + l3B + l4
    B, B1, B2, B3 = loghyp0f1(l, np.sqrt(hatCF*hatCG)*C/2/D)
    A = np.log(hyp0f1((2*l+1)/2, hatCF*hatCG*C**2/4/D**2))
    return np.sum(logp + B), np.sum(np.isfinite(logp + A))


generator = jointClGenCP('/Users/harrybevins/Documents/Software/cosmopower')
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt('cosmology-data/wmap_unbinned.txt', unpack=True)
lplanck, signal_planck, _, _ = np.loadtxt('cosmology-data/planck_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

min_l = np.arange(2, 1000, 25)
print(min_l)
ratio, l1, l2 = [], [], []
for i in range(len(min_l)):
    mask = lwmap > min_l[i]
    lwmap = lwmap[mask]
    bins = bins[mask]

    pnoise = planck_noise(lwmap).calculate_noise()
    wnoise = wmap_noise(lwmap).calculate_noise()

    planck_binned_like_wmap = binning(signal_planck, bins)
    wmap_binned_like_wmap = binning(wmap_unbinned, bins)

    samples = [0.022, 0.12, 0.96, 3.0448, 0.674]
    pobs, wobs, cltheory = generator(samples, lwmap, bins)

    cltheory = cltheory[0]
    pobs = pobs[0]
    wobs = wobs[0]

    like1 = loglikelihood(planck_binned_like_wmap + pnoise, 
                        wmap_binned_like_wmap + wnoise, 
                        cltheory, pnoise, wnoise, lwmap)
    like2 = loglikelihood(pobs, wobs, cltheory, 
                        pnoise, wnoise, lwmap)
    ratio.append(like2[0] - like1[0])
    l1.append(like1[0])
    l2.append(like2[0])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(min_l, ratio)
axes[0].set_xlabel('Minimum l')
axes[0].set_ylabel('LogL(Real Data) \n - LogL(Simulated Obs <-- Planck Params)')
axes[0].set_yscale('log')

cb = axes[1].scatter(l2, l1, c=min_l, cmap='viridis')
fig.colorbar(cb, ax=axes[1], label='Minimum l')
axes[1].set_xlabel('LogL(Simulated Obs <-- Planck Params)')
axes[1].set_ylabel('LogL(Real Data)')
axes[1].set_ylim(axes[1].get_xlim())
axes[1].plot(l2, l2, 'k--')
print(axes[1].get_xlim())

for i in range(len(l1)):
    print(l1[i], l2[i], min_l[i], axes[1].get_xlim()[0])
    if l1[i] < axes[1].get_xlim()[0]:
        axes[1].arrow(l2[i], axes[1].get_ylim()[0]*5, 0, 
                      -15, color='r',
                      head_width=5, head_length=3)

plt.tight_layout()
plt.savefig('min_l_vs_accuracy.png')
plt.show()
