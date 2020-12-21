# -*- coding: utf-8 -*-

from defs cimport *
from grid cimport GridData,GridCoord,BnzGrid
from particle cimport BnzParticles
from integrator cimport BnzIntegr

cdef class BnzIO

cdef void write_history(BnzIO, BnzGrid, BnzIntegr)
cdef void write_grid(BnzIO, BnzGrid, BnzIntegr, int)
cdef void write_slice(BnzIO, BnzGrid, BnzIntegr)
cdef void write_particles(BnzIO, BnzGrid, BnzIntegr, int)
