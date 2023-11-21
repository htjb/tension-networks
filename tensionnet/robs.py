import numpy as np
import matplotlib.pyplot as plt
#from mpi4py import MPI
from anesthetic import read_chains
import pypolychord
from pypolychord.settings import  PolyChordSettings
from anesthetic import make_2d_axes

def run_poly(prior, likelihood, file, RESUME = False, nDims=4,
             **kwargs):

    nlive = kwargs.get('nlive', None)
    nDerived = kwargs.get('nDerived', 0)

    settings = PolyChordSettings(nDims, 0) #settings is an object
    settings.read_resume = RESUME
    settings.base_dir = file + '/'
    if nlive:
        settings.nlive = nlive
   
    output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
    paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
    output.make_paramnames_files(paramnames)
