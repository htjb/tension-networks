import numpy as np
import camb
from scipy.stats import multivariate_normal
from pypolychord.priors import UniformPrior

class BAO():
    def __init__(self, **kwargs):
        
        self.pars = camb.CAMBparams()
        self.data_location = kwargs.pop('data_location', 'bao_data/')
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

class SDSS_BAO(BAO):
    def __init__(self, **kwargs):
        self.data_location = kwargs.pop('data_location', 'bao_data/')
        super().__init__(data_location=self.data_location)
        #self.d12, self.d12cov, self.d16, self.d16cov = \
        #      self.get_data()
        self.d12, self.d12cov = self.get_data()
        #self.z = np.hstack((self.d12[:, 0], self.d16[:, 0]))[::2]
        self.z = self.d12[:, 0]

    def get_data(self):
        d12 = np.loadtxt(self.data_location + 'sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0, 1])
        #d16 = np.loadtxt(self.data_location + 'sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0, 1])
        d12cov = np.loadtxt(self.data_location + 'sdss_DR12_LRG_BAO_DMDH_covtot.txt')
        #d16cov = np.loadtxt(self.data_location + 'sdss_DR16_LRG_BAO_DMDH_covtot.txt')
        return d12, d12cov,# d16, d16cov

    def get_camb_model(self, theta):
        self.pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=0.055, H0=100*theta[4])
        self.pars.InitPower.set_params(As=np.exp(theta[3])/10**10, ns=theta[2])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(self.pars) # computes evolution of background cosmology

        da = (1+self.z) * results.angular_diameter_distance(self.z)
        dh = 3e5/results.hubble_parameter(self.z) # 1/Mpc
        rs = results.get_derived_params()['rdrag'] # Mpc

        datad12 = [da[0]/rs, dh[0]/rs, da[1]/rs, dh[1]/rs]
        #datad16 = [da[2]/rs, dh[2]/rs]

        return datad12#, datad16
    
    def loglikelihood(self):
        def likelihood(theta):

            #datad12, datad16 = self.get_camb_model(theta)
            datad12 = self.get_camb_model(theta)

            L1 = multivariate_normal(mean=self.d12[:, 1], cov=self.d12cov).logpdf(datad12)
            #L2 = multivariate_normal(mean=self.d16[:, 1], cov=self.d16cov).logpdf(datad16)

            logl = L1# + L2
            return logl, []
        return likelihood
    
    def get_sample(self, theta):
        #datad12, datad16 = self.get_camb_model(theta)
        datad12 = self.get_camb_model(theta)

        noisey12 = multivariate_normal(mean=datad12, cov=self.d12cov).rvs()
        #noisey16 = multivariate_normal(mean=datad16, cov=self.d16cov).rvs()
        #return noisey12, noisey16, datad12, datad16
        return noisey12, datad12

class DESI_BAO(BAO):
    def __init__(self, **kwargs):
        self.data_location = kwargs.pop('data_location', 'bao_data/')
        super().__init__(data_location=self.data_location)
        #self.L1, self.L2, self.L1cov, self.L2cov = self.get_data()
        #self.z = np.array([self.L1[0, 0], self.L2[0, 0]])
        self.L2, self.L2cov = self.get_data()
        self.z = self.L2[:, 0]
    
    def get_data(self):
        #L1 = np.loadtxt(self.data_location + 
        #                'desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt', 
        #                usecols=[0, 1])
        L2 = np.loadtxt(self.data_location + 
                        'desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt',
                        usecols=[0, 1])
        #L1cov = np.loadtxt(self.data_location + 
        #                   'desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt')
        L2cov = np.loadtxt(self.data_location + 
                           'desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt')
        #return L1, L2, L1cov, L2cov
        return L2, L2cov
    
    def get_camb_model(self, theta):
        self.pars.set_cosmology(ombh2=theta[0], omch2=theta[1],
                            tau=0.055, H0=100*theta[4])
        self.pars.InitPower.set_params(As=np.exp(theta[3])/10**10, ns=theta[2])
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_background(self.pars)

        da = (1+self.z) * results.angular_diameter_distance(self.z)
        dh = 3e5/results.hubble_parameter(self.z)
        rs = results.get_derived_params()['rdrag']

        datal2 = [da[0]/rs, dh[0]/rs]
        #datal2 = [da[1]/rs, dh[1]/rs]

        #return datal1, datal2
        return datal2
    
    def loglikelihood(self):

        def likelihood(theta):

            #datal1, datal2 = self.get_camb_model(theta)
            datal2 = self.get_camb_model(theta)

            #Like1 = multivariate_normal(mean=self.L1[:, 1], cov=self.L1cov).logpdf(datal1)
            Like2 = multivariate_normal(mean=self.L2[:, 1], cov=self.L2cov).logpdf(datal2)

            #logl = Like1 + Like2
            logl = Like2
            return logl, []
        return likelihood
    
    def get_sample(self, theta):
        #datal1, datal2 = self.get_camb_model(theta)
        datal2 = self.get_camb_model(theta)

        #noiseyL1 = multivariate_normal(mean=datal1, cov=self.L1cov).rvs()
        noiseyL2 = multivariate_normal(mean=datal2, cov=self.L2cov).rvs()
        #return noiseyL1, noiseyL2, datal1, datal2
        return noiseyL2, datal2

