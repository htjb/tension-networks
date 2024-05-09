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

def loglikelihood(hatCF, hatCG, C, NF, NG, l, axes):
    """ takes in the observed power spectra, theory, noise and relevant l"""
    axes[0].plot(l, hatCF*l*(l+1)/2/np.pi, label='hatCF')
    axes[0].plot(l, hatCG*l*(l+1)/2/np.pi, label='hatCG')
    axes[0].plot(l, C*l*(l+1)/2/np.pi, label='Theory')
    axes[0].plot(l, NF*l*(l+1)/2/np.pi, label='NoiseF')
    axes[0].plot(l, NG*l*(l+1)/2/np.pi, label='NoiseG')
    axes[0].legend()

    axes[1].plot(l, hatCF*l*(l+1)/2/np.pi, label='hatCF')
    axes[1].plot(l, hatCG*l*(l+1)/2/np.pi, label='hatCG')
    axes[1].plot(l, C*l*(l+1)/2/np.pi, label='Theory')
    axes[1].plot(l, NF*l*(l+1)/2/np.pi, label='NoiseF')
    axes[1].plot(l, NG*l*(l+1)/2/np.pi, label='NoiseG')
    axes[1].set_xlim(-10, 100)
    axes[1].set_ylim(0, 1500)
    axes[1].legend()

    D = ((C+NF)*(C+NG) - C**2)/(2*l+1)
    l1 = -2*loggamma((2*l+1)/2) 
    l2 = - (2*l+1)/2*np.log(4*D/(2*l+1)) 
    l3A = - ((C+NG)*hatCF)/(2*D)
    l3B = -((C+NF)*hatCG)/(2*D)
    l4 = (2*l-1)/2*np.log(hatCF*hatCG)
    logp = l1 + l2 + l3A + l3B + l4
    B, B1, B2, B3 = loghyp0f1(l, np.sqrt(hatCF*hatCG)*C/2/D)

    flag = True
    if flag:
        delta = C.min()
        logpenalty = -2.5
        print(np.log(delta), logpenalty - np.log(delta))
        emax = logp+B > logpenalty - np.log(delta)
        logp = np.where(emax, logp+B, logpenalty - np.log(delta))
        axes[2].axhline(logpenalty - np.log(delta), color='k', lw=0.5)
    else:
        logp = logp+B

    #axes[2].plot(l, logp+A, marker='*', label='Scipy logL= {:.2f}'.format(np.sum(logp + A)))
    axes[2].plot(l, logp, marker='.', label='Will Approximation\nlogL= {:.2f}'.format(np.sum(logp)))
    axes[2].set_ylabel('Log Likelihood')
    for i in range(len(axes)):
        axes[i].set_xlabel('l')
    axes[0].set_ylabel('C_l')
    axes[1].set_ylabel('C_l')
    axes[2].legend()

    axes[3].plot(l, l1, label='l1')
    axes[3].plot(l, l2, label='l2')
    axes[3].plot(l, l3A, label='l3A (hatCF)')
    axes[3].plot(l, l3B, label='l3B (hatCG)')
    axes[3].plot(l, l4, label='l4 (hatCF, hatCG)')
    axes[3].plot(l, B, label='Will Approximation\n(hatCF, hatCG)')
    axes[3].plot(l, 1/D, label='1/D')
    axes[3].legend(fontsize=8)

    axes[4].plot(l, B+l3A+l3B, label='Will Approximation\n(hatCF, hatCG)\n+ l3A + l3B')
    axes[4].legend()

    return np.sum(logp)


generator = jointClGenCP('/Users/harrybevins/Documents/Software/cosmopower')
wmap_data = np.loadtxt('cosmology-data/wmap_binned.txt')
lwmap_raw, wmap_unbinned, _, _, _ = np.loadtxt('cosmology-data/wmap_unbinned.txt', unpack=True)
lplanck, signal_planck, _, _ = np.loadtxt('cosmology-data/planck_unbinned.txt', unpack=True)

bins = np.array([wmap_data[:, 1], wmap_data[:, 2]]).T
lwmap = wmap_data[:, 0]

mask = lwmap > 0
lwmap = lwmap[mask]
bins = bins[mask]

pnoise = planck_noise(lwmap).calculate_noise()
wnoise = wmap_noise(lwmap).calculate_noise()

planck_binned_like_wmap = binning(signal_planck, bins)
wmap_binned_like_wmap = binning(wmap_unbinned, bins)

#np.savetxt('cosmology-data/planck_binned_like_wmap.txt', planck_binned_like_wmap)

#samples = prior(np.random.uniform(0, 1, 5))
samples = [0.022, 0.12, 0.96, 3.0448, 0.674]
pobs, wobs, cltheory = generator(samples, lwmap, bins)

#from tensionnet.wmapplanck import loglikelihood

cltheory = cltheory[0]
pobs = pobs[0]
wobs = wobs[0]

#planck_binned_like_wmap = np.hstack([wmap_binned_like_wmap[lwmap <=500], 
#                                     planck_binned_like_wmap[lwmap >500]])

fig, axes = plt.subplots(2, 5, figsize=(15, 10))
like1 = loglikelihood(planck_binned_like_wmap + pnoise, 
                     wmap_binned_like_wmap + wnoise, 
                    cltheory, pnoise, wnoise, lwmap, axes[0, :])
like2 = loglikelihood(pobs, wobs, cltheory, 
                      pnoise, wnoise, lwmap, axes[1, :])
print(like1, like2)

axes[0, 1].set_title('True Observations')
axes[1, 1].set_title('Generated Observations (Approx. Planck Params)')
plt.tight_layout()
#plt.savefig('joint_likelihood_comparison_l' + str(lwmap.min()) + '_wmap_replace_planck_lowl.png')
plt.savefig('joint_likelihood_comparison_l' + str(lwmap.min()) + '.png')
plt.show()

"""fig, axes = plt.subplots(1, 2, figsize=(5, 3))
axes[0].plot(lwmap, np.log10(planck_binned_like_wmap/pnoise), label='Planck')
axes[0].plot(lwmap, np.log10(wmap_binned_like_wmap/wnoise), label='WMAP')
axes[0].axhline(0, lw=0.5, color='k')
axes[0].set_xlabel('l')
axes[0].set_ylabel('log10(SNR)')
axes[0].legend()

print(lwmap[wnoise>=1e-3].min())
print(lwmap[wnoise>=7e-4].min())
axes[1].plot(lwmap, pnoise, label='Planck')
axes[1].plot(lwmap, wnoise, label='WMAP')
axes[1].set_xlabel('l')
axes[1].set_ylabel('N_l')
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
plt.savefig('snr_comparison.png')
plt.show()"""