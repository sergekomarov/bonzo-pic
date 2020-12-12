# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.coord_cy cimport GridCoord

cdef extern from "particle.h" nogil:

  # Properties of a particle specie.

  ctypedef struct SpcProp:
    real qm     # charge-to-mass ratio
    long Np     # number of particles

  # General particle properties.

  ctypedef struct PrtProp:

    int ppc            # number of particles per cell
    int Nprop          # number of particle properties
    long Npmax         # length of particle array
    long Np            # number of active particles of all species
    int Ns             # number of species
    SpcProp *spc_props # properties of particle species

    real c             # speed of light / Courant number
    real me            # electron mass
    real mime          # ion-electron mass ratio
    # real c_ompe      # electron skin depth

  # Arrays of particle properties.

  ctypedef struct PrtData:

    # coordinates
    real *x
    real *y
    real *z

    # four-velocities
    real *u
    real *v
    real *w

    real *g     # relativistic gamma

    real *m     # mass
    int *spc    # specie
    long *id    # particle ID


from bnz.particle.prt_bc cimport PrtBc

# Particle class.

cdef class BnzParticles:

  cdef:
    PrtProp *prop
    PrtData *data
    PrtBc bc

  cdef bytes usr_dir
