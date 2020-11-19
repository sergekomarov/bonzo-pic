# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf

from bnz.utils cimport print_root



cdef void print_nrg(BnzSim sim):

  cdef:
    ints i,j,k, n
    int id
    real Ekm=0., Emm=0., Eem=0.

  cdef:
    GridParams gp = sim.grid.params
    GridData gd = sim.grid.data
    BnzParticles prts = sim.prts

    ints Ncells = gp.Nact_glob[0] * gp.Nact_glob[1] * gp.Nact_glob[2]

    real[::1] Em_loc = np.zeros(OMP_NT)
    real[::1] Ee_loc = np.zeros(OMP_NT)
    real[::1] Ek_loc = np.zeros(OMP_NT)

  cdef ints rank=0
  IF MPI:
    rank = mpi.COMM_WORLD.Get_rank()
    cdef:
      real[::1] var     = np.empty(3, dtype='f8')
      real[::1] var_sum = np.empty(3, dtype='f8')
      

  with nogil, parallel(num_threads=OMP_NT):
    id = threadid()

    for k in prange(gp.k1, gp.k2+1, schedule='dynamic'):
      for j in range(gp.j1, gp.j2+1):
        for i in range(gp.i1, gp.i2+1):

          Ee_loc[id] = Ee_loc[id] + 0.5*(gd.E[0,k,j,i]**2 + gd.E[1,k,j,i]**2 + gd.E[2,k,j,i]**2)
          Em_loc[id] = Em_loc[id] + 0.5*(gd.B[0,k,j,i]**2 + gd.B[1,k,j,i]**2 + gd.B[2,k,j,i]**2)


  with nogil, parallel(num_threads=OMP_NT):
    id = threadid()

    for n in prange(prts.Np, schedule='dynamic'):

      Ek_loc[id] = Ek_loc[id] + prts.data.m[n] * (prts.data.g[n]-1)


  for i in range(OMP_NT):

    Eem += Ee_loc[i]
    Emm += Em_loc[i]
    Ekm += Ek_loc[i]

  Eem /= Ncells
  Emm /= Ncells
  Ekm /= Ncells
  Ekm *= sim.phys.c**2

  IF MPI:

    var[0], var[1], var[2] = Eem, Emm, Ekm
    mpi.COMM_WORLD.Allreduce(var, var_sum, op=mpi.SUM)
    Eem, Emm, Ekm = var_sum[0], var_sum[1], var_sum[2]

  print_root(rank, "\n----- mean energy densities -------\n")
  print_root(rank, "Ek = %f\n", Ekm)
  print_root(rank, "Ee = %f\n", Eem)
  print_root(rank, "Em = %f\n", Emm)
  print_root(rank, "Etot = %f\n", Ekm+Eem+Emm)

  print_root(rank, "-----------------------------------\n")

  return
