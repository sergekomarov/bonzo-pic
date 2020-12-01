# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from bnz.io.read_config import read_param
from bnz.problem.problem cimport set_phys_funcs_user


# =======================================

# Integration-related data and functions.

cdef class BnzIntegr:

  def __cinit__(self, bytes usr_dir):

    # cdef:
    #   GridCoord gc = self.coord
    #   GridData gd = self.data

    # Courant number
    self.cour = read_param("computation", "cour", 'f', usr_dir)

    # Number of passes of current filter.
    self.Nfilt = read_param("computation", "Nfilt", 'i',usr_dir)

    # allocate arrays used by MHD integrator

    # sh_u = (NWAVES,gc.Ntot[2],gc.Ntot[1],gc.Ntot[0])
    # sh_3 = (3,Nz,Ny,Nx)

    # set pointers to user-defined physics functions (e.g. gravitational potential)
    set_phys_funcs_user(self)


  def __dealloc__(self):

    return
