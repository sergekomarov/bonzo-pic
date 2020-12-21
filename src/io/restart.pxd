# -*- coding: utf-8 -*-

from defs cimport *
from grid cimport GridCoord,GridData, BnzGrid
from particle cimport BnzParticles
from integrator cimport BnzIntegr

cdef class BnzIO

cdef void set_restart_grid(BnzIO, BnzGrid, BnzIntegr)
cdef void set_restart_particles(BnzIO, BnzGrid, BnzIntegr)
