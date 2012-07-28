from __future__ import division
from timeit import timeit

NUM_RUNS = 5
NSAMPLES = 500

aux_setup = \
'''
import sampling
import numpy as np
alpha = {alpha}
data = np.{data!r}
'''

mh_setup = aux_setup + '\nbeta = {beta}\n'

def get_auxvar_timing(**kwargs):
    stmt = \
            '''
            sampling.generate_pi_samples_withauxvars(alpha,%d,data)
            ''' % NSAMPLES
    return timeit(stmt=stmt,setup=aux_setup.format(**kwargs),number=NUM_RUNS)/NUM_RUNS/NSAMPLES

def get_mh_timing(**kwargs):
    stmt = \
            '''
            sampling.generate_pi_samples_mh(alpha,%d,data,beta)
            ''' % NSAMPLES
    return timeit(stmt=stmt,setup=mh_setup.format(**kwargs),number=NUM_RUNS)/NUM_RUNS/NSAMPLES

