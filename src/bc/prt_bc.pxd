# -*- coding: utf-8 -*-

from defs cimport *
from grid cimport GridCoord, GridData
from particle cimport PrtProp, PrtData
from integrator cimport BnzIntegr

# Particle boundary conditions.

# particle BC function pointer
ctypedef void (*PrtBcFunc)(PrtData*,PrtProp*, GridData,GridCoord, BnzIntegr)

cdef class PrtBC:

  # BC flags
  cdef int bc_flags[3][2]

  # BC function pointers
  cdef PrtBcFunc prt_bc_funcs[3][2]

  # BC buffers
  real2d sendbuf, recvbuf
  long recvbuf_size, sendbuf_size

  cdef void apply_prt_bc(self, PrtData*,PrtProp*, GridData,GridCoord, BnzIntegr)
