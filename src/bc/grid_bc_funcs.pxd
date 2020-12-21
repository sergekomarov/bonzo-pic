# -*- coding: utf-8 -*-

from defs cimport *
from grid cimport GridData,GridCoord
from integrator cimport BnzIntegr

cdef void x1_grid_bc_periodic(GridData,GridCoord, BnzIntegr, int1d)
cdef void x2_grid_bc_periodic(GridData,GridCoord, BnzIntegr, int1d)
cdef void y1_grid_bc_periodic(GridData,GridCoord, BnzIntegr, int1d)
cdef void y2_grid_bc_periodic(GridData,GridCoord, BnzIntegr, int1d)
cdef void z1_grid_bc_periodic(GridData,GridCoord, BnzIntegr, int1d)
cdef void z2_grid_bc_periodic(GridData,GridCoord, BnzIntegr, int1d)

cdef void x1_grid_bc_outflow(GridData,GridCoord, BnzIntegr, int1d)
cdef void x2_grid_bc_outflow(GridData,GridCoord, BnzIntegr, int1d)
cdef void y1_grid_bc_outflow(GridData,GridCoord, BnzIntegr, int1d)
cdef void y2_grid_bc_outflow(GridData,GridCoord, BnzIntegr, int1d)
cdef void z1_grid_bc_outflow(GridData,GridCoord, BnzIntegr, int1d)
cdef void z2_grid_bc_outflow(GridData,GridCoord, BnzIntegr, int1d)

cdef void x1_grid_bc_conduct(GridData,GridCoord, BnzIntegr, int1d)
cdef void x2_grid_bc_conduct(GridData,GridCoord, BnzIntegr, int1d)
cdef void y1_grid_bc_conduct(GridData,GridCoord, BnzIntegr, int1d)
cdef void y2_grid_bc_conduct(GridData,GridCoord, BnzIntegr, int1d)
cdef void z1_grid_bc_conduct(GridData,GridCoord, BnzIntegr, int1d)
cdef void z2_grid_bc_conduct(GridData,GridCoord, BnzIntegr, int1d)

IF MPI:
  cdef void pack_grid_all(GridData,GridCoord, int1d, real1d, int,int)
  cdef void unpack_grid_all(GridData,GridCoord, int1d, real1d, int,int)
