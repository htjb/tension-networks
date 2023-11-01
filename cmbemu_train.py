from cmbemu.preprocess import process
from pypolychord.priors import UniformPrior
from cmbemu.network import nn
import numpy as np
import matplotlib.pyplot as plt
import camb
from tqdm import tqdm

GEN_DATA = False
PREPROCESS = False
TRAIN = True
data_dir = 'cmbemu_training_data/'
model_dir = 'cmbemu_model/'
l = np.linspace(2, 2000, 300)

# generate test and train data
if GEN_DATA:
    def prior():
        theta = np.zeros((10000, 6))
        theta[:, 0] = np.random.uniform(0.0211, 0.0235, 10000) # omegabh2
        theta[:, 1] = np.random.uniform(0.108, 0.131, 10000) # omegach2
        theta[:, 2] = np.random.uniform(1.038, 1.044, 10000) # 100*thetaMC
        theta[:, 3] = np.random.uniform(0.01, 0.16, 10000) # tau
        theta[:, 4] = np.random.uniform(0.938, 1, 10000) # ns
        theta[:, 5] = np.random.uniform(2.95, 3.25, 10000) # log(10^10*As)
        return theta
    
    theta = prior()
    pars = camb.CAMBparams()

    cls = []
    for i in tqdm(range(len(theta))):
        t = theta[i].copy()
        try:
            pars.set_cosmology(ombh2=t[0], omch2=t[1],
                            tau=t[3], cosmomc_theta=t[2]/100)
            pars.InitPower.set_params(As=np.exp(t[5])/10**10, ns=t[4])
            results = camb.get_background(pars) # computes evolution of background cosmology

            cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:, 0]
            cls.append(np.interp(l, np.arange(len(cl)), cl))
        except:
            print('CAMB failed')
            pass

    cls = np.array(cls)
    from sklearn.model_selection import train_test_split

    train_data, test_data, train_labels, test_labels = train_test_split(theta, cls, test_size=0.2)

    import os
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    np.savetxt(data_dir+'train_data.txt', train_data)
    np.savetxt(data_dir+'train_labels.txt', train_labels)
    np.savetxt(data_dir+'test_data.txt', test_data)
    np.savetxt(data_dir+'test_labels.txt', test_labels)

# preprocess and train
if PREPROCESS:
    process('full', l, base_dir=model_dir, data_location=data_dir)
if TRAIN:
    nn(batch_size=300, epochs=1000, base_dir=model_dir, layer_sizes=[14, 14, 14, 14],
        input_shape=7, patience=100, lr=1e-3)

test_d = np.loadtxt(data_dir + 'test_data.txt')
test_l = np.loadtxt(data_dir + 'test_labels.txt')

from cmbemu.eval import evaluate

predictor = evaluate(base_dir=model_dir)

for i in range(len(test_d)):
   if i%250 == 0:
      emu, l = predictor(test_d[i])
      plt.plot(l, emu)
      plt.plot(l, test_l[i])
      plt.show()

error = []
for i in range(len(test_d)):
    error.append(np.abs(predictor(test_d[i])[0] - test_l[i])/test_l[i]*100)
error = np.array(error)

plt.plot(l, np.mean(error, axis=0))
plt.show()
