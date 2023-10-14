import numpy as np
import matplotlib.pyplot as plt

exp1_data_no_tension = np.loadtxt('test_case_chains/exp1_data_no_tension.txt')
exp2_data_no_tension = np.loadtxt('test_case_chains/exp2_data_no_tension.txt')
exp1_data_tension = np.loadtxt('test_case_chains/exp1_data_in_tension.txt')
exp2_data_tension = np.loadtxt('test_case_chains/exp2_data_in_tension.txt')

exp1_freq = np.linspace(60, 90, 100)
exp2_freq = np.linspace(80, 120, 100)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(exp1_freq, exp1_data_no_tension, label='Exp 1')
axes[0].plot(exp2_freq, exp2_data_no_tension, label='Exp 2')
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Signal')    
axes[0].legend()
axes[0].set_title('No Tension')

axes[1].plot(exp1_freq, exp1_data_tension, label='Exp 1')
axes[1].plot(exp2_freq, exp2_data_tension, label='Exp 2')
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Signal')
axes[1].legend()
axes[1].set_title('In Tension')

plt.tight_layout()
plt.savefig('test_case_data.png', dpi=300)
plt.show()

