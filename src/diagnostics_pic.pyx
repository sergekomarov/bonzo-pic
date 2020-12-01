# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdio cimport printf

from bnz.utils cimport print_root, sqr


cdef void print_nrg(BnzGrid grid, BnzIntegr integr):

  cdef:
    ints i,j,k, n
    int id
    real ekm=0., emm=0., eem=0.

  cdef:
    GridCoord gc = grid.coord
    GridData gd = grid.data
    BnzParticles prts = grid.prts

    ints ncells = gc.Nact_glob[0] * gc.Nact_glob[1] * gc.Nact_glob[2]

    real1d em_loc = np.zeros(OMP_NT)
    real1d ee_loc = np.zeros(OMP_NT)
    real1d ek_loc = np.zeros(OMP_NT)

  IF MPI:
    cdef:
      double[::1] var     = np.empty(3, dtype='f8')
      double[::1] var_sum = np.empty(3, dtype='f8')


  with nogil, parallel(num_threads=OMP_NT):
    id = threadid()

    for k in prange(gc.k1, gc.k2+1, schedule='dynamic'):
      for j in range(gc.j1, gc.j2+1):
        for i in range(gc.i1, gc.i2+1):

          ee_loc[id] = ee_loc[id] + 0.5*(sqr(gd.ee[0,k,j,i]) + sqr(gd.ee[1,k,j,i]) + sqr(gd.ee[2,k,j,i]))
          em_loc[id] = em_loc[id] + 0.5*(sqr(gd.bf[0,k,j,i]) + sqr(gd.bf[1,k,j,i]) + sqr(gd.bf[2,k,j,i]))


  with nogil, parallel(num_threads=OMP_NT):
    id = threadid()

    for n in prange(prts.prop.Np, schedule='dynamic'):

      ek_loc[id] = ek_loc[id] + prts.data.m[n] * (prts.data.g[n]-1.)


  for i in range(OMP_NT):

    eem += ee_loc[i]
    emm += em_loc[i]
    ekm += ek_loc[i]

  eem /= ncells
  emm /= ncells
  ekm /= ncells
  ekm *= integr.cour**2

  IF MPI:

    var[0], var[1], var[2] = eem, emm, ekm
    mpi.COMM_WORLD.Allreduce(var, var_sum, op=mpi.SUM)
    eem, emm, ekm = var_sum[0], var_sum[1], var_sum[2]

  print_root("\n----- mean energy densities -------\n")
  print_root("Ek = %f\n", ekm)
  print_root("Ee = %f\n", eem)
  print_root("Em = %f\n", emm)
  print_root("Etot = %f\n", ekm+eem+emm)

  print_root("-----------------------------------\n")

  return
