# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
import sys

from libc.stdlib cimport free, calloc

from bnz.utils cimport free_3d_array, calloc_3d_array, print_root
from bnz.io.read_config import read_param
from hilbertcurve import HilbertCurve

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


#-----------------------------------------------------------

cdef class GridCoord:

  def __cinit__(self, bytes usr_dir):

    # box size in cells
    self.Nact_glob[0] = read_param("computation","Nx",'i',usr_dir)
    self.Nact_glob[1] = read_param("computation","Ny",'i',usr_dir)
    self.Nact_glob[2] = read_param("computation","Nz",'i',usr_dir)

    cdef int ndim=1
    IF D2D: ndim += 1
    IF D3D: ndim += 1

    if ndim==1:
      if self.Nact_glob[1] != 1:
        print_root('Error: cannot set Ny>1 in 1D.')
        sys.exit()

    if ndim==2:
      if self.Nact_glob[2] != 1:
        print_root('Error: cannot set Nz>1 in 2D/1D.')
        sys.exit()

    # Set the number of ghost cells.

    Nfilt = read_param("computation", "Nfilt", 'i',usr_dir)
    self.ng = IMIN(4, IMAX(1, Nfilt))

    self.Ntot_glob[0] = self.Nact_glob[0] + 2*self.ng+1
    IF D2D: self.Ntot_glob[1] = self.Nact_glob[1] + 2*self.ng+1
    ELSE:   self.Ntot_glob[1] = 1
    IF D3D: self.Ntot_glob[2] = self.Nact_glob[2] + 2*self.ng+1
    ELSE:   self.Ntot_glob[2] = 1

    for n in range(3):
      self.lmin[n] = 0.
      self.lmax[n] = <real>self.Ntot_glob[n]

    # Set min/max indices of active cells.
    # (will be reset when domain decomposition is used)

    self.i1, self.i2 = self.ng, self.Nact_glob[0] + self.ng - 1

    IF D2D: self.j1, self.j2 = self.ng, self.Nact_glob[1] + self.ng - 1
    ELSE: self.j1, self.j2 = 0,0

    IF D3D: self.k1, self.k2 = self.ng, self.Nact_glob[2] + self.ng - 1
    ELSE: self.k1, self.k2 = 0,0

    # Set the same local size as global for now.
    # (will be reset when domain decomposition is used)

    for k in range(3):
      self.Nact[k] = self.Nact_glob[k]
      self.Ntot[k] = self.Ntot_glob[k]

    self.rank=0
    for k in range(3):
      self.pos[k]=0
      self.size[k]=1
      self.size_tot=1
    self.ranks=NULL


# -----------------------------------------------------------

cdef class GridData:

  def __cinit__(self, GridCoord gc):

    #cdef GridScratch scr =   grid.scr

    sh_3 = (3, gc.Ntot[2], gc.Ntot[1], gc.Ntot[0])

    gd.efld = np.zeros(sh_3, dtype=np_real)
    gd.bfld = np.zeros(sh_3, dtype=np_real)
    gd.curr = np.zeros(sh_3, dtype=np_real)

    # IF MPI:
    #   scr.ce_tmp = np.zeros(sh_3, dtype=np_real)


# ----------------------------------------------------------

cdef class BnzGrid:

  def __cinit__(self, bytes usr_dir):

    self.usr_dir = usr_dir

    self.coord = GridCoord(usr_dir)

    # set index-related parameters
    init_params(self.coord, usr_dir)

    # do domain decomposition
    IF MPI: domain_decomp(self.coord, usr_dir)

    # init boundary conditions
    self.bc = GridBc(self.coord, usr_dir)

    # init data arrays
    self.data = GridData(self.coord)

    # init particles
    self.prts = BnzParticles(self.coord, usr_dir)

  def __dealloc__(self):

    IF MPI: free_3d_array(self.coord.ranks)


# ----------------------------------------------------------------------

IF MPI:

  cdef void domain_decomp(GridCoord *gc, bytes usr_grid):

    cdef mpi.Comm comm = mpi.COMM_WORLD

    cdef:
      int i,k, size0
      int p
      long[:,::1] crds

    # read number of MPI blocks in each direction
    gc.size[0] = read_param("computation", "nblocks_x", 'i', usr_dir)
    gc.size[1] = read_param("computation", "nblocks_y", 'i', usr_dir)
    gc.size[2] = read_param("computation", "nblocks_z", 'i', usr_dir)

    gc.rank = comm.Get_rank()
    size0 = comm.Get_size()
    gc.size_tot = gc.size[0] * gc.size[1] * gc.size[2]

    # check if number of blocks is consistent with number of processes and grid size

    if gc.size_tot != size0:
      print_root("Total number of MPI blocks is not equal to number of processors!\n")
      sys.exit()

    for k in range(3):
      if gc.Nact_glob[k] % gc.size[k] != 0:
        print_root("Number of MPI blocks is not a multiple of grid size in %i-direction!\n", k)
        sys.exit()

    hilbert_curve = None

    # distribute MPI blocks across processors

    if gc.size[1]>1 and gc.size[2]>1:

      # check if numbers of blocks in each direction are same and equal to 2**p
      if gc.size[0]==gc.size[1] and gc.size[1]==gc.size[2] and (gc.size[0] & (gc.size[0] - 1)) == 0:

        print_root('Using 2d Hilbert-curve domain decomposition...\n')

        p=<int>np.log2(gc.size[0])

        # generate a 3D Hilbert curve
        hilbert_curve = HilbertCurve(p,3)

        crd = hilbert_curve.coordinates_from_distance(gc.rank)
        gc.pos[0] = crd[0]
        gc.pos[1] = crd[1]
        gc.pos[2] = crd[2]

        # print gc.id, crd

    elif gc.size[1]>1:

      if gc.size[0]==gc.size[1] and (gc.size[0] & (gc.size[0] - 1)) == 0:

        print_root('using 3d Hilbert-curve domain decomposition...\n')

        p=<int>np.log2(gc.size[0])

        # generate a 2D Hilbert curve
        hilbert_curve = HilbertCurve(p,2)

        crd = hilbert_curve.coordinates_from_distance(gc.rank)
        gc.pos[0] = crd[0]
        gc.pos[1] = crd[1]
        gc.pos[2] = 0

        # print gc.id, crd

    # if not successful, use the simplest possible arrangement of blocks

    if hilbert_curve==None:

      gc.pos[2] =  gc.rank / (gc.size[0] * gc.size[1])
      gc.pos[1] = (gc.rank % (gc.size[0] * gc.size[1])) / gc.size[0]
      gc.pos[0] = (gc.rank % (gc.size[0] * gc.size[1])) % gc.size[0]

    # gather positions of all MPI blocks
    crds = np.empty((gc.size_tot, 3), dtype=np.int_)  # int_ is same as C long
    comm.Allgather([np.array(gc.pos), mpi.LONG], [crds, mpi.LONG])

    gc.ranks = <int ***>calloc_3d_array(gc.size[0], gc.size[1], gc.size[2], sizeof(int))

    for i in range(gc.size_tot):
      gc.ranks[crds[i,0], crds[i,1], crds[i,2]] = i

    # if gc.rank==0:
    #   np.save('ids.npy', np.asarray(gc.ranks))

    # for i in range(gc.size[0]):
    #   for j in range(gc.size[1]):
    #     for m in range(gc.size[2]):
    #       gc.ranks[i,j,m] = i * gc.size[1] * gc.size[2] + j * gc.size[2] + m


    # save IDs of neighboring blocks

    gc.nbr_ranks[0][0] = gc.ranks[gc.pos[0]-1, gc.pos[1],   gc.pos[2]]   if gc.pos[0] != 0 else -1
    gc.nbr_ranks[0][1] = gc.ranks[gc.pos[0]+1, gc.pos[1],   gc.pos[2]]   if gc.pos[0] != gc.size[0]-1 else -1
    gc.nbr_ranks[1][0] = gc.ranks[gc.pos[0],   gc.pos[1]-1, gc.pos[2]]   if gc.pos[1] != 0 else -1
    gc.nbr_ranks[1][1] = gc.ranks[gc.pos[0],   gc.pos[1]+1, gc.pos[2]]   if gc.pos[1] != gc.size[1]-1 else -1
    gc.nbr_ranks[2][0] = gc.ranks[gc.pos[0],   gc.pos[1],   gc.pos[2]-1] if gc.pos[2] != 0 else -1
    gc.nbr_ranks[2][1] = gc.ranks[gc.pos[0],   gc.pos[1],   gc.pos[2]+1] if gc.pos[2] != gc.size[2]-1 else -1

    # print rank, gc.x1nbr_id, gc.xid, gc.x2nbr_id
    # print rank, gc.y1nbr_id, gc.yid, gc.y2nbr_id
    # print rank, gc.z1nbr_id, gc.zid, gc.z2nbr_id

    for k in range(3): gc.Nact[k] /= gc.size[k]

    gc.Ntot[0] = gc.Nact[0] + 2*gc.ng + 1
    IF D2D:
      gc.Ntot[1] = gc.Nact[1] + 2*gc.ng + 1
    ELSE:
      gc.Ntot[1] = 1
    IF D3D:
      gc.Ntot[2] = gc.Nact[2] + 2*gc.ng + 1
    ELSE:
      gc.Ntot[2] = 1

    gc.i1, gc.i2 = gc.ng, gc.Nact[0] + gc.ng - 1

    IF D2D: gc.j1, gc.j2 = gc.ng, gc.Nact[1] + gc.ng - 1
    ELSE:   gc.j1, gc.j2 = 0,0

    IF D3D: gc.k1, gc.k2 = gc.ng, gc.Nact[2] + gc.ng - 1
    ELSE:   gc.k1, gc.k2 = 0,0


# end of IF MPI
