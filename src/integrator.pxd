# -*- coding: utf-8 -*-
from bnz.defs_cy cimport *

cdef class BnzIntegr:

  cdef:
    real time
    real tmax
    long step
    real dt

  # Courant number
  cdef real cour

  # number of passes of current filter
  cdef int nfilt
