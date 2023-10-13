import numpy as np
from edges_saras.data_gen import saras3_func_gen, edges_func_gen
from edges_saras.priors import saras_foreground_prior, edges_foreground_prior, signal_prior
import matplotlib.pyplot as plt
from edges_saras.observational_data.interface import get_observational_data
from tensorflow.keras.optimizers.schedules import ExponentialDecay

"""lr_schedule = ExponentialDecay(
                    initial_learning_rate=1e-4,
                    decay_steps=1000,
                    decay_rate=0.95,
                )"""

edges = get_observational_data('EDGES')
saras = get_observational_data('SARAS3')
edges_freq = edges['Frequency [MHz]'].values
saras3_freq = saras['Frequency [MHz]'].values

saras3_freq = np.linspace(saras3_freq.min(), saras3_freq.max(), 100)
edges_freq = np.linspace(edges_freq.min(), edges_freq.max(), 100)

saras = saras3_func_gen(saras3_freq)
edges = edges_func_gen(edges_freq)

from tensionnet.tensionnet import nre

nrei = nre(lr=1e-4)
nrei.build_model(len(saras3_freq) + len(edges_freq), 1, [100]*5, 'relu')
#nrei.default_nn_model(len(saras3_freq) + len(edges_freq))
#nrei.build_compress_model(len(saras3_freq), len(edges_freq), 1, 
#                       [len(saras3_freq), len(saras3_freq), len(saras3_freq)//2, 50, 10], 
#                       [len(edges_freq), len(edges_freq), len(edges_freq)//2, 50, 10], 
#                       [10, 10, 10, 10, 10],
#                       'relu')
nrei.build_simulations(saras, edges, saras_foreground_prior, edges_foreground_prior, signal_prior, n=500000)
model, data_test, labels_test = nrei.training(epochs=1000, batch_size=1000)

plt.plot(nrei.loss_history)
plt.plot(nrei.test_loss_history)
plt.show()

nrei.__call__()
print(nrei.r_values)
plt.hist(nrei.r_values, bins=50)
plt.show()