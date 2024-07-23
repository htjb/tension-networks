import numpy as np
from tensionnet.bao import SDSS_BAO, DESI_BAO
import matplotlib.pyplot as plt
from tensionnet.utils import plotting_preamble

plotting_preamble()

desi_baos = DESI_BAO(data_location='cosmology-data/bao_data/')
sdss_baos = SDSS_BAO(data_location='cosmology-data/bao_data/')

desiLRG, desiLRGcov, desiLRGELG, desiLRGELGcov, desiELG, desiELGcov = desi_baos.get_data()
sdss, sdsscov = sdss_baos.get_data()

fig, axes = plt.subplots(1, 1, figsize=(3.5, 2.5))
axes.errorbar(desiLRG[0, 0], desiLRG[0, 1], 
              yerr=np.sqrt(np.diag(desiLRGcov))[0], markersize=4.5,
              marker='x', c='C0')
axes.errorbar(desiLRG[1, 0], desiLRG[1, 1], 
              yerr=np.sqrt(np.diag(desiLRGcov))[1], markersize=4.5,
              marker='x', c='C1')

axes.errorbar(desiLRGELG[0, 0], desiLRGELG[0, 1], 
              yerr=np.sqrt(np.diag(desiLRGELGcov))[0], markersize=4.5,
              marker='o', c='C0')
axes.errorbar(desiLRGELG[1, 0], desiLRGELG[1, 1], 
              yerr=np.sqrt(np.diag(desiLRGELGcov))[1], markersize=4.5,
              marker='o', c='C1')

axes.errorbar(desiELG[0, 0], desiELG[0, 1], 
              yerr=np.sqrt(np.diag(desiELGcov))[0], markersize=4.5,
              marker='^', c='C0')
axes.errorbar(desiELG[1, 0], desiELG[1, 1], 
              yerr=np.sqrt(np.diag(desiELGcov))[1], markersize=4.5,
              marker='^', c='C1')

axes.errorbar(sdss[0, 0], sdss[0, 1], 
              yerr=np.sqrt(np.diag(sdsscov))[0], markersize=4.5,
              marker='x', c='C0')
axes.errorbar(sdss[1, 0], sdss[1, 1], 
              yerr=np.sqrt(np.diag(sdsscov))[1], markersize=4.5,
                marker='x', c='C1')
axes.errorbar(sdss[2, 0], sdss[2, 1], markersize=4.5,
              yerr=np.sqrt(np.diag(sdsscov))[2], marker='x', c='C0')
axes.errorbar(sdss[3, 0], sdss[3, 1], markersize=4.5,
              yerr=np.sqrt(np.diag(sdsscov))[3], marker='x', c='C1')

axes.plot(1.4, 18, 'x', label='LRG', c='k')
axes.plot(1.4, 18, 'o', label='LRG + ELG', c='k')
axes.plot(1.4, 18, '^', label='ELG', c='k')
axes.plot([1.4, 1.5], [18, 18], label=r'$D_M/r_s$')
axes.plot([1.4, 1.5], [18, 18], label=r'$D_H/r_s$')

axes.axvspan(0.2, 0.6, color='C2', alpha=0.5) # SDSS LRG
axes.axvspan(0.6, 1.35, color='C3', alpha=0.5) # DESI LRG + ELG
axes.text(0.4, 27, 'SDSS', color='k', fontsize=12,
          bbox={'facecolor': 'white', 'pad': 5})
axes.text(0.9, 27, 'DESI', color='k', fontsize=12,
          bbox={'facecolor': 'white', 'pad': 5})

axes.set_xlim(0.35, 1.35)

plt.xlabel(r'$z$')
plt.ylabel(r'$D/r_s$')
plt.legend(loc='upper left', bbox_to_anchor=(0.05, -0.2), ncols=2)
#plt.tight_layout()
plt.savefig('desi_sdss_bao.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
