import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

scale = 1
x = np.random.normal(5, scale, 10000)
x = np.sort(x)
n = norm(loc=5, scale=scale)
p = n.cdf(x)

example_r = 5.87

fig, axes = plt.subplots(3, 2, figsize=(5, 4))

# plotting the histrograms and the cdfs
for i in range(2):
    axes[0, i].hist(x, bins=30, density=True)
    axes[0, i].axvline(example_r, color='C4', ls='--')
    axes[0, i].set_xlabel(r'$\log R$')
    axes[0, i].set_ylabel(r'$P(\log R)$')
    axes[0, i].set_xlim(0, 8)

    axes[1, i].plot(x, p)
    axes[1, i].axhline(n.cdf(example_r), color='C4', ls='--')
    axes[1, i].set_xlabel(r'$\log R$')
    axes[1, i].set_ylabel(r'$P(\log R < \log R^\prime)$')
     
# plotting the confidence regions
axes[1, 0].axhspan(1-0.68, 1, color='C1', alpha=0.8)
axes[1, 0].axhspan(1-0.95, 1, color='C1', alpha=0.5)
axes[1, 0].axhspan(1-0.98, 1, color='C1', alpha=0.2)

axes[1, 1].axhspan(0, 0.68, color='C1', alpha=0.8)
axes[1, 1].axhspan(0, 0.95, color='C1', alpha=0.5)
axes[1, 1].axhspan(0, 0.98, color='C1', alpha=0.2)

# plotting the  tension statistic
y = norm.isf(p/2)
axes[2, 0].plot(p, norm.isf(p/2))
#axes[2, 0].plot(p, norm.ppf(1 - p/2))
axes[2, 0].axvline(n.cdf(example_r), color='C4', ls='--')
axes[2, 0].axhline(norm.isf(n.cdf(example_r)/2), color='C4', 
                   ls='--', label=r'$T =$' + f'{norm.isf(n.cdf(example_r)/2):.2f}')
axes[2, 0].axvspan(np.interp(1, y[::-1], p[::-1]), 1, color='C1', alpha=0.8)
axes[2, 0].axvspan(np.interp(2, y[::-1], p[::-1]), 1, color='C1', alpha=0.5)
axes[2, 0].axvspan(np.interp(3, y[::-1], p[::-1]), 1, color='C1', alpha=0.3)
#axes[2, 0].set_xscale('log')
axes[2, 0].set_xlabel(r'$P(\log R < \log R^\prime)$')
axes[2, 0].set_ylabel(r'$T$')
axes[2, 0].legend(fontsize=8)

# plottign the agreement statistic
y = norm.isf((1-p)/2)
axes[2, 1].plot(p, norm.isf((1-p)/2))
#axes[2, 1].plot(p, norm.ppf(1 - (1-p)/2), c='r')
axes[2, 1].axvline(n.cdf(example_r), color='C4', ls='--')
axes[2, 1].axhline(norm.isf((1 - n.cdf(example_r))/2), color='C4', 
                   ls='--', label=r'$C =$' + f'{norm.isf((1 - n.cdf(example_r))/2):.2f}')

axes[2, 1].axvspan(np.interp(1, y, p), 0, color='C1', alpha=0.8)
axes[2, 1].axvspan(np.interp(2, y, p), 0, color='C1', alpha=0.5)
axes[2, 1].axvspan(np.interp(3, y, p), 0, color='C1', alpha=0.3)
#axes[2, 0].set_xscale('log')
axes[2, 1].set_xlabel(r'$P(\log R < \log R^\prime)$')
axes[2, 1].set_ylabel(r'$C$')
axes[2, 1].legend(fontsize=8)

axes[0, 0].set_title('How in tension are\nthe data sets?')
axes[0, 1].set_title('How consistent are\nthe data sets?')
#axes[0, 2].set_title('How unexpected is the\nrecovered value of ' + r'$\log R$?')

plt.tight_layout()
plt.savefig('figures/figure2.pdf', dpi=300, bbox_inches='tight')
plt.show()

