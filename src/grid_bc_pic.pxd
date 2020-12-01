# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *

cdef class BnzSim

# grid BC function pointer
ctypedef void (*GridBcFunc)(BnzSim, ints[::1])

# Boundary condition class.

cdef class GridBc:

  # BC flags
  # 0 - periodic; 1 - outflow; 2 - conductive; 3 - user-defined
  cdef int bc_flags[3][2]

  # array of grid BC function pointers
  cdef GridBcFunc grid_bc_funcs[3][2]

  IF MPI:
    cdef:
      real2d sendbuf, recvbuf    # send/receive buffers for boundary conditions
      ints recvbuf_size, sendbuf_size   # buffer sizes


cdef void apply_grid_bc(BnzSim, ints[::1])
