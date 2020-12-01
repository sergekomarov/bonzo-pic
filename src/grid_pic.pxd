# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *


ctypedef struct GridCoord:

  # Cells and indices.

  ints Nact[3]        # numbers of active cells cells on local grid
  ints Ntot[3]        # numbers of cells including ghosts
  ints Nact_glob[3]   # active cells in full domain
  ints Ntot_glob[3]   # all cells in full domain
  ints ng             # number of ghost cells
  ints i1,i2          # min and max indices of active cells on local grid
  ints j1,j2
  ints k1,k2

  # Coordinates.

  real lmin[3]        # coordinates of left border of global domain
  real lmax[3]        # coordinates of right border of global domain

  # MPI block IDs
  ints rank             # MPI rank of the grid
  ints pos[3]           # 3D index of the grid on the current processor

  ints size[3]         # number of MPI blocks (grids) in x,y,z directions
  ints size_tot        # total number of blocks

  ints ***ranks        # 3D array of grid ranks
  ints nbr_ranks[3][2] # ranks of neighboring grids
  # nbr_ids[axis,L(0)/R(1)]


#=======================================================================

# All grid data.

cdef class GridData:

  cdef:
    real4d ee            # edge-centered electric field
    real4d bf            # face-centered magnetic field
    real4d ce            # edge-centered currents


#=======================================================================

# Scratch arrays used by the integrator (mainly by diffusion routines).

cdef class GridScratch:

  cdef real4d ce_tmp      # temporary copy of the particle current array


cdef class GridBc

# Grid class, contains parameters and data of local grid.

cdef class BnzGrid:

  cdef:
    GridCoord coord      # grid coordinates
    GridData data        # grid data
    GridScratch scratch  # scratch arrays
    GridBc bc            # boundary conditions
    BnzParticles prts    # particles

  cdef bytes usr_dir     # user directory, contains config file

  cdef void init(self)
