import numpy as np
import matplotlib.pyplot as plt

base_dir = 'toy_chains_temp_sweep/'
exp1_data = np.loadtxt(base_dir + 'exp1_data_no_tension.txt')

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)

temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45]
temps = np.sort(temps)
for i, t in enumerate(temps):
    exp2_data_no_tension = np.loadtxt(base_dir + f'exp2_data_{t}.txt')
    plt.plot(exp2_freq, exp2_data_no_tension, label=f'Exp. 2: {t} K', c='r', alpha=1/(i+1))

plt.plot(exp1_freq, exp1_data, label='Exp. 1: 0.2 K', c='k')
plt.legend()
plt.xlabel('Frequency [MHz]')
plt.ylabel(r'$\delta T_b$ [K]')
plt.tight_layout()
plt.savefig('test_case_data.png', dpi=300)
plt.show()

