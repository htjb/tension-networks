import numpy as np
from tensionnet.bao import SDSS_BAO, DESI_BAO
import matplotlib.pyplot as plt
from tensionnet.utils import plotting_preamble

plotting_preamble()

desi_baos = DESI_BAO(data_location='cosmology-data/bao_data/')
sdss_baos = SDSS_BAO(data_location='cosmology-data/bao_data/')

desiLRG, desiLRGcov, desiLRGELG, desiLRGELGcov, desiELG, desiELGcov = desi_baos.get_data()
sdss, sdsscov = sdss_baos.get_data()

fig, axes = plt.subplots(1, 1, figsize=(6.3, 3))
axes.errorbar(desiLRG[0, 0], desiLRG[0, 1], 
              yerr=np.sqrt(np.diag(desiLRGcov))[0], markersize=4.5,
              marker='o', c='C0', label='DESI LRG ' + r'$D_M/r_s$')
axes.errorbar(desiLRG[1, 0], desiLRG[1, 1], 
              yerr=np.sqrt(np.diag(desiLRGcov))[1], markersize=4.5,
              marker='o', c='C1', label='DESI LRG ' + r'$D_H/r_s$')

axes.errorbar(desiLRGELG[0, 0], desiLRGELG[0, 1], 
              yerr=np.sqrt(np.diag(desiLRGELGcov))[0], markersize=4.5,
              marker='^', c='C0', label='DESI LRG+ELG ' + r'$D_M/r_s$')
axes.errorbar(desiLRGELG[1, 0], desiLRGELG[1, 1], 
              yerr=np.sqrt(np.diag(desiLRGELGcov))[1], markersize=4.5,
              marker='^', c='C1', label='DESI LRG+ELG ' + r'$D_H/r_s$')

axes.errorbar(desiELG[0, 0], desiELG[0, 1], 
              yerr=np.sqrt(np.diag(desiELGcov))[0], markersize=4.5,
              marker='x', c='C0', label='DESI ELG ' + r'$D_M/r_s$')
axes.errorbar(desiELG[1, 0], desiELG[1, 1], 
              yerr=np.sqrt(np.diag(desiELGcov))[1], markersize=4.5,
              marker='x', c='C1', label='DESI ELG ' + r'$D_H/r_s$')

axes.errorbar(sdss[0, 0], sdss[0, 1], 
              yerr=np.sqrt(np.diag(sdsscov))[0], markersize=4.5,
              marker='*', c='C0', label='SDSS LRG ' + r'$D_M/r_s$')
axes.errorbar(sdss[1, 0], sdss[1, 1], 
              yerr=np.sqrt(np.diag(sdsscov))[1], markersize=4.5,
                marker='*', c='C1', label='SDSS LRG ' + r'$D_H/r_s$')
axes.errorbar(sdss[2, 0], sdss[2, 1], markersize=4.5,
              yerr=np.sqrt(np.diag(sdsscov))[2], marker='*', c='C0')
axes.errorbar(sdss[3, 0], sdss[3, 1], markersize=4.5,
              yerr=np.sqrt(np.diag(sdsscov))[3], marker='*', c='C1')


#axes.axvspan(0.2, 0.6, color='C2', alpha=0.5) # SDSS LRG
#axes.axvspan(0.6, 1.35, color='C3', alpha=0.5) # DESI LRG + ELG

axes.set_xlim(0.35, 1.35)

plt.xlabel(r'$z$')
plt.ylabel(r'$D/r_s$')
plt.legend(loc='lower left', bbox_to_anchor=(-0.15, 1.05), ncols=3)
plt.tight_layout()
plt.savefig('desi_sdss_bao.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
