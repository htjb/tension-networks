import numpy as np
import camb
from tqdm import tqdm
from pypolychord.priors import UniformPrior
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def wide_prior(cube):
    # wide prior apart from tau which I left tight
    theta = np.zeros(len(cube))
    theta[0] = UniformPrior(0.01, 0.085)(cube[0]) # omegabh2
    theta[1] = UniformPrior(0.08, 0.21)(cube[1]) # omegach2
    theta[2] = UniformPrior(0.97, 1.5)(cube[2]) # 100*thetaMC
    theta[3] = UniformPrior(0.01, 0.16)(cube[3]) # tau
    theta[4] = UniformPrior(0.8, 1.2)(cube[4]) # ns
    theta[5] = UniformPrior(2.6, 3.8)(cube[5]) # log(10^10*As)
    return theta

pars = camb.CAMBparams()

a = 1
nsamples = 100000

perrank = nsamples//size
base_dir = 'cl_library/'

if rank == 0:
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

comm.Barrier()

cls, params = [], []
for i in tqdm(range(a + rank*perrank, a + (rank+1)*perrank)):
    parameters = wide_prior(np.random.rand(6))
    pars.set_cosmology(ombh2=parameters[0], omch2=parameters[1],
                        tau=parameters[3], cosmomc_theta=parameters[2]/100,
                        theta_H0_range=[5, 1000])
    pars.InitPower.set_params(As=np.exp(parameters[5])/10**10, ns=parameters[4])
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_background(pars) # computes evolution of background cosmology

    cl = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total'][:,0]
    #cl = np.interp(l, np.arange(len(cl)), cl)
    cl *= 2*np.pi/(np.arange(len(cl))*(np.arange(len(cl))+1))
    cl = cl[1:]
    cls.append(cl)
    params.append(parameters)

cls = np.array(cls)
params = np.array(params)

np.savetxt(base_dir + 'cls_{}.txt'.format(rank), cls)
np.savetxt(base_dir + 'params_{}.txt'.format(rank), params)
