import numpy as np
import camb
from scipy.stats import multivariate_normal
from pypolychord.priors import UniformPrior

class BAO():
    def __init__(self, **kwargs):
        
        self.pars = camb.CAMBparams()
        self.data_location = kwargs.pop('data_location', 'bao_data/')
        self.d12, self.d16, self.d12cov, self.d16cov = \
              self.get_data(data_location=self.data_location)
        self.z = np.hstack((self.d12[:, 0], self.d16[:, 0]))[::2]
        self.prior_mins = kwargs.pop('prior_mins', [0.01, 0.08, 0.8, 2.6, 0.5])
        self.prior_maxs = kwargs.pop('prior_maxs', [0.085, 0.21, 1.2, 3.8, 0.9])

    def prior(self, cube):

        """
        Prior on the cosmological parameters 
        modified from https://arxiv.org/abs/1902.04029.

        Parameters
        ----------
        cube: array
            Array of values between 0 and 1.
        
        Returns
        -------
        theta: array
            Array of cosmological parameters.

        """

        theta = np.zeros(len(cube))
        theta[0] = UniformPrior(self.prior_mins[0], 
                                self.prior_maxs[0])(cube[0]) # omegabh2
        theta[1] = UniformPrior(self.prior_mins[1], 
                                self.prior_maxs[1])(cube[1]) # omegach2
        theta[2] = UniformPrior(self.prior_mins[2], 
                                self.prior_maxs[2])(cube[2]) # ns
        theta[3] = UniformPrior(self.prior_mins[3], 
                                self.prior_maxs[3])(cube[3]) # log(10^10*As)
        theta[4] = UniformPrior(self.prior_mins[4], 
                                self.prior_maxs[4])(cube[4]) # H0
        return theta
    
    def get_camb_model(self, theta):
        self.pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=0.055,
                            theta_H0_range=[5, 1000], H0=100*theta[4])
        self.pars.InitPower.set_params(As=np.exp(theta[3])/10**10, ns=theta[2])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(self.pars) # computes evolution of background cosmology

        da = (1+self.z) * results.angular_diameter_distance(self.z)
        dh = 3e5/results.hubble_parameter(self.z) # 1/Mpc
        rs = results.get_derived_params()['rdrag'] # Mpc

        datad12 = [da[0]/rs, dh[0]/rs, da[1]/rs, dh[1]/rs]
        datad16 = [da[2]/rs, dh[2]/rs]

        return datad12, datad16
    
    def get_data(self, data_location='bao_data/'):
        d12 = np.loadtxt(data_location + 'sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
        d16 = np.loadtxt(data_location + 'sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])
        d12cov = np.loadtxt(data_location + 'sdss_DR12_LRG_BAO_DMDH_covtot.txt')
        d16cov = np.loadtxt(data_location + 'sdss_DR16_LRG_BAO_DMDH_covtot.txt')
        return d12, d16, d12cov, d16cov
    
    def loglikelihood(self):
        def likelihood(theta):

            datad12, datad16 = self.get_camb_model(theta)

            L1 = multivariate_normal(mean=self.d12[:, 1], cov=self.d12cov).logpdf(datad12)
            L2 = multivariate_normal(mean=self.d16[:, 1], cov=self.d16cov).logpdf(datad16)

            logl = L1 + L2
            return logl, []
        return likelihood
    
    def get_sample(self, theta):
        datad12, datad16 = self.get_camb_model(theta)

        noisey12 = multivariate_normal(mean=datad12, cov=self.d12cov).rvs()
        noisey16 = multivariate_normal(mean=datad16, cov=self.d16cov).rvs()
        return noisey12, noisey16, datad12, datad16