import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from pypolychord.priors import UniformPrior
import numpy as np
import matplotlib.pyplot as plt
import pickle

def simulation(parameters):
    return [torch.normal(parameters[0], 0.5), torch.normal(parameters[1], 0.5)]

true_signal = torch.tensor([5, 7])

nDims = 2
rounds = 2
num_sims=1000

try:
    posterior = pickle.load(open('gaussian_param_posterior.pkl', 'rb'))
except FileNotFoundError:
    prior = utils.BoxUniform(low=torch.tensor([0, 0]),
                                high=torch.tensor([10, 10]))

    posterior = infer(simulation, prior, method="SNLE", num_simulations=1000)

    with open('gaussian_param_posterior.pkl', 'wb') as f:
        pickle.dump(posterior, f)

print(posterior)

samples = posterior.sample((100,), x=true_signal)
log_probability = posterior.log_prob(samples, x=true_signal)
_ = analysis.pairplot(samples, figsize=(6, 6))
plt.show()

def gaussian_likelihood(parameters):
    return (-0.5*np.log(2*np.pi*0.5**2) - 0.5*(5 -parameters[0])**2/0.5**2) + \
        (-0.5*np.log(2*np.pi*0.5**2) - 0.5*(7 - parameters[1])**2/0.5**2)


logL = np.array([gaussian_likelihood(samples[i].numpy()) for i in range(len(samples))])
prior = np.log(1/10*1/10)

plt.plot(log_probability.numpy()-prior, logL, 'o')
plt.plot(logL, logL, 'k--')
plt.ylabel('log likelihood')
plt.xlabel('log probability')
plt.title('Gaussian likelihood vs. NLE')
plt.savefig('gaussian_likelihood_vs_nle.png')
plt.show()