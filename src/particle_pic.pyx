# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from libc.stdlib cimport free, calloc
from bnz.utils cimport mini,maxi
from bnz.io.read_config import read_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzParticles:

  def __cinit__(self, GridCoord *gc, bytes usr_dir):

    self.usr_dir = usr_dir

    # self.prop = ParticleProp()  # structure
    # self.data = ParticleData()  # structure

    cdef ParticleProp *pp = &(self.prop)

    # number of particles per cell
    pp.ppc = <ints>read_param("computation","ppc",'i',usr_dir)

    IF D2D and D3D:
      pp.ppc = (<ints>(pp.ppc**(1./3)))**3
    ELIF D2D:
      pp.ppc = (<ints>sqrt(pp.ppc))**2

    # ppc needs to be even to have equal numbers of electrons and ions
    if pp.ppc % 2 != 0: pp.ppc += 1

    pp.c = read_param("computation", "cour", 'f',usr_dir)
    pp.mime = read_param("physics",  "mime", 'f',usr_dir)
    cdef real c_ompe = read_param("physics", "c_ompe", 'f',usr_dir)
    pp.me = pp.c / (c_ompe * pp.ppc)

    pp.Np = pp.ppc * gc.Nact[0] * gc.Nact[1] * gc.Nact[2]

    # Set properties of particle species.

    pp.Ns=2    # 0: ions; 1: electrons by default
    pp.spc_props = <SpcProp*>calloc(pp.Ns, sizeof(SpcProp))

    pp.spc_props[0].qm =  1./pp.mime
    pp.spc_props[1].qm = -1.
    pp.spc_props[0].Np=  <ints>(0.5*pp.Np)
    pp.spc_props[1].Np=  <ints>(0.5*pp.Np)
    pp.Nprop = 10


  cdef void init(self, GridCoord *gc):

    self.bc = ParticleBC(self.prop, gc, self.usr_dir)
    init_data(self)


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


# ------------------------------------------------------------------------

cdef void init_data(BnzParticles prts):

  # Call AFTER domain decomposition.

  cdef:
    ParticleProp *pp = &(prts.prop)
    ParticleData *pd = &(prts.data)

  pp.Npmax = pp.Np      # only when there is no injection of particles !!!
  IF MPI: pp.Npmax = <ints>(1.3*pp.Npmax)

  pd.x = <real *>calloc(pp.Npmax, sizeof(real))
  pd.y = <real *>calloc(pp.Npmax, sizeof(real))
  pd.z = <real *>calloc(pp.Npmax, sizeof(real))

  pd.u = <real *>calloc(pp.Npmax, sizeof(real))
  pd.v = <real *>calloc(pp.Npmax, sizeof(real))
  pd.w = <real *>calloc(pp.Npmax, sizeof(real))
  pd.g = <real *>calloc(pp.Npmax, sizeof(real))

  pd.m = <real *>calloc(pp.Npmax, sizeof(real))
  pd.spc = <ints *>calloc(pp.Npmax, sizeof(ints))
  pd.id = <ints *>calloc(pp.Npmax, sizeof(ints))
