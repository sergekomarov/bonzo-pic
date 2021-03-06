# -*- coding: utf-8 -*-

from defs cimport *


cdef class GridCoord:

  # Cells and indices.

  int Nact[3]        # numbers of active cells cells on local grid
  int Ntot[3]        # numbers of cells including ghosts
  int Nact_glob[3]   # active cells in full domain
  int Ntot_glob[3]   # all cells in full domain
  int ng             # number of ghost cells
  int i1,i2          # min and max indices of active cells on local grid
  int j1,j2
  int k1,k2

  # Coordinates.

  real lmin[3]        # coordinates of left border of global domain
  real lmax[3]        # coordinates of right border of global domain

  # MPI data.
  int rank            # MPI rank of the grid
  int pos[3]          # 3D index of the grid on the current processor
  int size[3]         # number of MPI blocks (grids) in x,y,z directions
  int size_tot        # total number of blocks
  int ***ranks        # 3D array of grid ranks
  int nbr_ranks[3][2] # ranks of neighboring grids
  # nbr_ids[axis,L(0)/R(1)]


cdef class GridData:

  cdef:
    real4d efld            # edge-centered electric field
    real4d bfld            # face-centered magnetic field
    real4d curr            # edge-centered currents


# circular import: can use forward declarations instead?
# from bnz.bc.grid_bc cimport GridBc
# from bnz.particle.particle cimport BnzParticles
cdef class GridBc
cdef class BnzParticles
cdef class BnzIntegr

cdef class BnzGrid:

  cdef:
    GridCoord coord      # grid coordinates
    GridData data        # grid data
    GridBc bc            # boundary conditions
    BnzParticles prts    # particles

  cdef str usr_dir     # user directory, contains config file

  cdef void apply_bc(self, BnzIntegr, int1d)
  cdef void apply_prt_bc(self, BnzIntegr)
