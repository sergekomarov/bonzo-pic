# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.data_struct cimport *

cdef void deposit_curr(real4d, BnzParticles, GridParams, double) nogil
cdef void filter_curr(real4d, BnzSim, GridParams, int)
