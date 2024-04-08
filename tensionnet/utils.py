import numpy as np
from scipy.stats import norm

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
