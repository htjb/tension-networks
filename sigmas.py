import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ecdf, norm

x = np.hstack([np.random.normal(5, 3, 10000),
                np.random.normal(0.2, 1, 10000)])

fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].hist(x, bins=30, density=True)
ax[0].set_xlabel(r'$\log R$')
ax[0].set_ylabel(r'$P(\log R)$')
ax[0].axvline(np.mean(x), color='C4', ls='--', label='Mean: {:.2f}'.format(np.mean(x) ))
ax[0].axvline(np.median(x), color='r', ls='--', label='Median: {:.2f}'.format(np.median(x)))

hist, bins = np.histogram(x, bins=30, density=True)
centers = (bins[1:] + bins[:-1])/2
ax[0].axvline(centers[np.argmax(hist)], color='C1', ls='--', label='Mode: {:.2f}'.format(centers[np.argmax(hist)]))
ax[0].legend()

r = np.sort(x)
c = ecdf(x)

ax[1].plot(r, c.cdf.evaluate(r))
ax[1].axhline(c.cdf.evaluate(np.mean(x)), color='C4', ls='--', label='Mean: {:.2f}'.format(c.cdf.evaluate(np.mean(x))))
ax[1].axhline(c.cdf.evaluate(centers[np.argmax(hist)]), color='C1', ls='--', label='Mode: {:.2f}'.format(c.cdf.evaluate(centers[np.argmax(hist)])))
ax[1].axhline(c.cdf.evaluate(np.median(x)), color='r', ls='--', label='Median: {:.2f}'.format(c.cdf.evaluate(np.median(x))))
ax[1].set_xlabel(r'$\log R$')
ax[1].set_ylabel(r'$P(\log R < \log R_\mathrm{obs})$')
ax[1].legend()

sigmaD = norm.isf(c.cdf.evaluate(x)/2)
sigmaR = norm.isf((2 - 2*c.cdf.evaluate(x))/2)
args = np.argsort(sigmaD)
sigmaR = sigmaR[args]
sigmaD = sigmaD[args]

print(norm.isf(c.cdf.evaluate(np.median(x))/2))
print(norm.isf((2 - 2*c.cdf.evaluate(np.median(x)))/2))

print(sigmaD[np.where(np.isclose(sigmaR, 0))])
ax[2].plot(sigmaR, sigmaD, ls='--')
ax[2].axvline(norm.isf((2 - 2*c.cdf.evaluate(np.mean(x)))/2), color='C4', ls='--', label=r'$\sigma_R(Mean) =$' + f'{norm.isf((2 - 2*c.cdf.evaluate(np.mean(x)))/2):.2f}')
ax[2].axhline(norm.isf(c.cdf.evaluate(centers[np.argmax(hist)])/2), color='C1', ls='--', label=r'$\sigma_D(Mode) =$' + f'{norm.isf(c.cdf.evaluate(centers[np.argmax(hist)])/2):.2f}')
ax[2].axhline(norm.isf(c.cdf.evaluate(np.mean(x))/2), color='C4', ls='--', label=r'$\sigma_D(Mean) =$' + f'{norm.isf(c.cdf.evaluate(np.mean(x))/2):.2f}')
ax[2].axvline(norm.isf((2 - 2*c.cdf.evaluate(centers[np.argmax(hist)]))/2), color='C1', ls='--', label=r'$\sigma_R(Mode) =$' + f'{norm.isf((2 - 2*c.cdf.evaluate(centers[np.argmax(hist)]))/2):.2f}')
ax[2].axhline(norm.isf(c.cdf.evaluate(np.median(x))/2), color='r', ls='--', label=r'$\sigma_D(Median) =$' + f'{norm.isf(c.cdf.evaluate(np.median(x))/2):.2f}')
ax[2].axvline(norm.isf((2 - 2*c.cdf.evaluate(np.median(x)))/2), color='r', ls='--', label=r'$\sigma_R(Median) =$' + f'{norm.isf((2 - 2*c.cdf.evaluate(np.median(x)))/2):.2f}')
ax[2].legend()
ax[2].set_xlabel(r'$\sigma_R$')
ax[2].set_ylabel(r'$\sigma_D$')
plt.tight_layout()
plt.show()