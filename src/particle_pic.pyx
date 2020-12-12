# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport free, calloc
from bnz.io.read_config import read_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzParticles:

  def __cinit__(self, GridCoord gc, bytes usr_dir):

    # Set particle properties.

    self.usr_dir = usr_dir

    cdef:
      PrtProp *pp = self.prop
      PrtData  *pd = self.data

    # number of particles per cell
    pp.ppc = <int>read_param("computation","ppc",'i',usr_dir)

    IF D2D and D3D:
      pp.ppc = (<int>(pp.ppc**(1./3)))**3
    ELIF D2D:
      pp.ppc = (<int>sqrt(pp.ppc))**2

    # ppc needs to be even to have equal numbers of electrons and ions
    if pp.ppc % 2 != 0: pp.ppc += 1

    # speed of light / Courant number
    pp.c = read_param("computation", "cour", 'f',usr_dir)
    # ion/electron mass ratio
    pp.mime = read_param("physics",  "mime", 'f',usr_dir)
    # electron skin depth
    cdef real c_ompe = read_param("physics", "c_ompe", 'f',usr_dir)
    # electron mass (taking qe/me=1)
    pp.me = pp.c / (c_ompe * pp.ppc/2)

    # total number of particles
    pp.Np = <long>pp.ppc * gc.Nact[0] * gc.Nact[1] * gc.Nact[2]

    # Set properties of particle species.

    # number of species
    pp.Ns=2    # 0: ions; 1: electrons

    pp.spc_props = <SpcProp*>calloc(pp.Ns, sizeof(SpcProp))

    # charge-to-mass ratios
    pp.spc_props[0].qm =  1./pp.mime
    pp.spc_props[1].qm = -1.
    # particle numbers
    pp.spc_props[0].Np=  pp.Np/2
    pp.spc_props[1].Np=  pp.Np/2
    pp.Nprop = 10

    # Init data structures.

    pp.Npmax = pp.Np      # only when there is no injection of particles !!!
    IF MPI: pp.Npmax = <long>(1.3*pp.Npmax)

    pd.x = <real *>calloc(pp.Npmax, sizeof(real))
    pd.y = <real *>calloc(pp.Npmax, sizeof(real))
    pd.z = <real *>calloc(pp.Npmax, sizeof(real))

    pd.u = <real *>calloc(pp.Npmax, sizeof(real))
    pd.v = <real *>calloc(pp.Npmax, sizeof(real))
    pd.w = <real *>calloc(pp.Npmax, sizeof(real))
    pd.g = <real *>calloc(pp.Npmax, sizeof(real))

    pd.m = <real *>calloc(pp.Npmax, sizeof(real))
    pd.spc = <int *>calloc(pp.Npmax, sizeof(int))
    pd.id = <long *>calloc(pp.Npmax, sizeof(long))

    # Init boundary conditions.

    self.bc = PrtBc(pp, gc, usr_dir)


  def __dealloc__(self):

    # Free array of particle species.

    free(self.prop.spc_props)

    # Free particle data.

    free(self.data.x)
    free(self.data.y)
    free(self.data.z)

    free(self.data.u)
    free(self.data.v)
    free(self.data.w)
    free(self.data.g)

    free(self.data.id)
    free(self.data.spc)
    free(self.data.m)
