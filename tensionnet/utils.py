import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
from pypolychord.priors import UniformPrior, LogUniformPrior

def calcualte_stats(Rs, sigma_Rs, c):
    """
    Function to calcualte the tension statistics from the NRE CDF.

    Parameters:
    -----------
    Rs: float
        The observed tension value.
    sigma_Rs: float
        The error on the observed tension value.
    c: scipy.stats.ecdf
        The empirical CDF of the tension values predicted by the NRE.
    """

    sigmaD = norm.isf(c.cdf.evaluate(Rs)/2)
    sigma_D_upper = norm.isf((c.cdf.evaluate(Rs + sigma_Rs))/2)
    sigma_D_lower = norm.isf((c.cdf.evaluate(Rs - sigma_Rs))/2)
    sigmaA = norm.isf((1- c.cdf.evaluate(Rs))/2)
    sigma_A_upper = norm.isf((1 - c.cdf.evaluate(Rs + sigma_Rs))/2)
    sigma_A_lower = norm.isf((1 - c.cdf.evaluate(Rs - sigma_Rs))/2)
    p = 2 - 2*c.cdf.evaluate(Rs)
    sigmaR = norm.isf(p/2)
    sigmaR_upper = norm.isf((2 - 2*(c.cdf.evaluate(Rs + sigma_Rs)))/2)
    sigmaR_lower = norm.isf((2 - 2*(c.cdf.evaluate(Rs - sigma_Rs)))/2)
    return sigmaD, sigma_D_upper, sigma_D_lower, \
        sigmaA, sigma_A_upper, sigma_A_lower, \
            sigmaR, sigmaR_upper, sigmaR_lower

def twentyone_example(exp1_data, exp2_data, exp1_freq, exp2_freq):
    def signal_func_gen(freqs):
        def signal(parameters):
            amp, nu_0, w = parameters
            return -amp * np.exp(-(freqs-nu_0)**2 / (2*w**2))
        return signal

    def signal_poly_prior(cube):
        theta = np.zeros(4)
        theta[0] = UniformPrior(0, 4)(cube[0]) #amp
        theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
        theta[2] = UniformPrior(5, 40)(cube[2]) #w
        theta[3] = LogUniformPrior(0.001, 0.1)(cube[3]) #sigma
        return theta

    def joint_prior(cube):
        theta = np.zeros(5)
        theta[0] = UniformPrior(0, 4)(cube[0]) #amp
        theta[1] = UniformPrior(60, 90)(cube[1]) #nu_0
        theta[2] = UniformPrior(5, 40)(cube[2]) #w
        theta[3] = LogUniformPrior(0.001, 0.1)(cube[3]) #sigma1
        theta[4] = LogUniformPrior(0.001, 0.1)(cube[4]) #sigma2
        return theta

    def exp1likelihood(theta):
        # gaussian log-likelihood
        return (-0.5 * np.log(2*np.pi*theta[-1]**2) \
            - 0.5 * (exp1_data - signal_func_gen(exp1_freq)(theta[:-1]))**2\
                /theta[-1]**2).sum(),[]

    def exp2likelihood(theta):
        # gaussian log-likelihood
        return (-0.5 * np.log(2*np.pi*theta[-1]**2) \
            - 0.5 * (exp2_data - signal_func_gen(exp2_freq)(theta[:-1]))**2/\
                theta[-1]**2).sum(),[]

    def jointlikelihood(theta):
        return exp1likelihood(theta[:-1])[0] + \
            exp2likelihood([*theta[:-2], theta[-1]])[0], []
    
    return signal_poly_prior, \
        joint_prior, exp1likelihood, exp2likelihood, jointlikelihood

def rebin(signal, bins, weights=None):
    indices = bins - 2
    binned_signal = []
    for i in range(len(indices)):
        if indices[i, 0] == indices[i, 1]:
            binned_signal.append(signal[int(indices[i, 0])])
        else:
            if weights is None:
                binned_signal.append(
                    np.mean(signal[int(indices[i, 0]):int(indices[i, 1])+1]))
            else:
                binned_signal.append(
                    np.average(signal[int(indices[i, 0]):int(indices[i, 1])+1], 
                    weights=weights[int(indices[i, 0]):int(indices[i, 1])+1]))
    return np.array(binned_signal)

def cosmopower_prior():
    parameters = ['omegabh2', 'omegach2', 'ns', 'As', 'h']
    prior_mins = [0.005, 0.001, 0.7, 1.61, 0.2]
    prior_maxs = [0.04, 0.99, 1.3, 5, 1.0]
    return parameters, prior_mins, prior_maxs