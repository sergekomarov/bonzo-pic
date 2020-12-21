# -*- coding: utf-8 -*-

from defs cimport *
from grid cimport GridCoord,GridData
from integrator cimport BnzIntegr

# grid BC function pointer
ctypedef void (*GridBcFunc)(GridData,GridCoord*, BnzIntegr, int1d)

# Boundary condition class.

cdef class GridBc:

  # BC flags
  # 0 - periodic; 1 - outflow; 2 - conductive; 3 - user-defined
  cdef int bc_flags[3][2]

  # array of grid BC function pointers
  cdef GridBcFunc grid_bc_funcs[3][2]

  real2d sendbuf, recvbuf    # send/receive buffers for boundary conditions
  long recvbuf_size, sendbuf_size   # buffer sizes

  cdef void apply_grid_bc(self, GridData,GridCoord*, BnzIntegr, int1d)
