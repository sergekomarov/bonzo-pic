# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from utils_bc cimport *

IF MPI:
  IF SPREC:
    mpi_real = mpi.FLOAT
  ELSE:
    mpi_real = mpi.DOUBLE


# ==============================================================================

cdef list get_bvar_fld_list(GridData gd, int1d bvars):

  # Form a list of references to 3D arrays that need to be updated.

  cdef int bvar
  flds=[]

  for bvar in bvars:

    if bvar>=EX and bvar<=EZ:
      flds.append(gd.ee[bvar,...])

    elif bvar>=BX and bvar<=BZ:
      flds.append(gd.bf[bvar-3,...])

    elif bvar>=JX and bvar<=JZ:
      flds.append(gd.ce[bvar-6,...])

  return flds



# ==============================================================================

# Periodic MHD BC.

cdef void x1_grid_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1, i2=gc.i2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_x(flds[n], gc.i1-gc.ng, gc.i2-gc.ng+1, gc.ng, gc.Ntot)


cdef void x2_grid_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1, i2=gc.i2
    int nbvar=bvars.size
  ng = ng+1

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_x(flds[n], i2+1, i1, ng, gc.Ntot)


cdef void y1_grid_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1, j2=gc.j2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_y(flds[n], j1-ng, j2-ng+1, ng, gc.Ntot)


cdef void y2_grid_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1, j2=gc.j2
    int nbvar=bvars.size
  ng = ng+1

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_y(flds[n], j2+1, j1, ng, gc.Ntot)


cdef void z1_grid_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1, k2=gc.k2
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_z(flds[n], k1-ng, k2-ng+1, ng, gc.Ntot)


cdef void z2_grid_bc_periodic(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1, k2=gc.k2
    int nbvar=bvars.size
  ng = ng+1

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    copy_layer_z(flds[n], k2+1, k1, ng, gc.Ntot)



# ==========================================================================

# Outflow BC.

cdef void x1_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    prolong_x(flds[n], 0, i1-1, ng, gc.Ntot)


cdef void x2_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i2=gc.i2
    int nbvar=bvars.size
  ng=ng+1

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # if bvar_types[n]==CC_FLD:
    prolong_x(flds[n], 1, i2+1, ng, gc.Ntot)

    # elif bvar_types[n]== FC_FLD or bvar_types[n]==EC_FLD:
    #   prolong_x(flds[n], 1, i2+2, ng-1, gc.Ntot)


cdef void y1_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j1=gc.j1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    prolong_y(flds[n], 0, j1-1, ng, gc.Ntot)


cdef void y2_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, j2=gc.j2
    int nbvar=bvars.size
  ng=ng+1

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # if bvar_types[n]==CC_FLD:
    prolong_y(flds[n], 1, j2+1, ng, gc.Ntot)

    # elif bvar_types[n]== FC_FLD or bvar_types[n]==EC_FLD:
    #   prolong_y(flds[n], 1, j2+2, ng-1, gc.Ntot)


cdef void z1_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k1=gc.k1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):
    prolong_z(flds[n], 0, k1, ng, gc.Ntot)


cdef void z2_grid_bc_outflow(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, k2=gc.k2
    int nbvar=bvars.size
  ng=ng+1

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # if bvar_types[n]==CC_FLD:
    prolong_z(flds[n], 1, k2+1, ng, gc.Ntot)

    # elif bvar_types[n]== FC_FLD or bvar_types[n]==EC_FLD:
    #   prolong_z(flds[n], 1, k2+2, ng-1, gc.Ntot)



# ==========================================================================

# Conductive BC for data grids.

cdef void x1_grid_bc_conduct(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, ng=gc.ng, i1=gc.i1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    # set tangential E (1,2 components) to zero in ghost region
    if bvars[n]==EY or bvars[n]==EZ:
      set_layer_x(flds[n], 0, i1-ng, ng+1, gc.Ntot)

    # if bvars[n]==JX:
    #   # reflect currents with respect to boundary
    #   # change sign of normal current
    #   copy_reflect_layer_x(flds[n], -1, i1-ng, i1, ng, gc.Ntot)
    #
    # if bvars[n]==JY or bvars[n]==JZ:
    #   # keep sign of transverse currents
    #   copy_reflect_layer_x(flds[n],  1, i1-ng, i1+1, ng, gc.Ntot)



cdef void x2_grid_bc_conduct(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, i2=gc.i2, ng=gc.ng+1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==EY or bvars[n]==EZ:
      set_layer_x(flds[n], 0, i2+1, ng, gc.Ntot)

    # if bvars[n]==JX:
    #   copy_reflect_layer_x(flds[n], -1, i2+1, i2-ng+1, ng, gc.Ntot)
    #
    # if bvars[n]==JY or bvars[n]==JZ:
    #   copy_reflect_layer_x(flds[n], 1, i2+2, i2-ng+1, ng-1, gc.Ntot)


#-------------------------------------------------------------------------------


cdef void y1_grid_bc_conduct(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, j1=gc.j1, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==EX or bvars[n]==EZ:
      set_layer_y(flds[n], 0, j1-ng, ng+1, gc.Ntot)

    # if bvars[n]==JX or bvars[n]==JZ:
    #   copy_reflect_layer_y(flds[n], 1, j1-ng, j1+1, ng, gc.Ntot)
    #
    # if bvars[n]==JY
    #   copy_reflect_layer_y(flds[n], -1, j1-ng, j1, ng, gc.Ntot)


cdef void y2_grid_bc_conduct(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, j2=gc.j2, ng=gc.ng+1
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==EX or bvars[n]==EZ:
      set_layer_y(flds[n], 0, j2+1, ng, gc.Ntot)

    # if bvars[n]==JX or bvars[n]==JZ:
    #   copy_reflect_layer_y(flds[n], 1, j2+2, j2-ng+1, ng-1, gc.Ntot)
    #
    # if bvars[n]==JY:
    #   copy_reflect_layer_y(flds[n],...], -1, j2+1, j2-ng+1, ng, gc.Ntot)


#-------------------------------------------------------------------------------


cdef void z1_grid_bc_conduct(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, k1=gc.k1, ng=gc.ng
    int nbvar=bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==EX or bvars[n]==EY:
      set_layer_z(flds[n], 0, k1-ng, ng+1, gc.Ntot)

    # if bvars[n]==JX or bvars[n]==JY:
    #   copy_reflect_layer_z(flds[n], 1, k1-ng, k1+1, ng, gc.Ntot)
    #
    # if bvars[n]==JZ
    #   copy_reflect_layer_z(flds[n], -1, k1-ng, k1, ng, gc.Ntot)


cdef void z2_grid_bc_conduct(GridData gd, GridCoord *gc,  BnzIntegr integr, int1d bvars):

  cdef:
    int n, k2=gc.k2, ng=gc.ng+1
    int nbvar = bvars.size

  flds = get_bvar_fld_list(gd, bvars)

  for n in range(nbvar):

    if bvars[n]==EX or bvars[n]==EY:
      set_layer_z(flds[n], 0, k2+1, ng, gc.Ntot)

    # if bvars[n]==JX or bvars[n]==JY:
    #   copy_reflect_layer_z(flds[n], 1, k2+2, k2-ng+1, ng-1, gc.Ntot)
    #
    # if bvars[n]==JZ:
    #   copy_reflect_layer_z(flds[n], -1, k2+1, k2-ng+1, ng, gc.Ntot)


# ---------------------------------------------------------------------------

IF MPI:

  cdef void pack_grid_all(GridData gd, GridCoord *gc,  int1d bvars, real1d buf, int ax, int side):

    cdef:
      int n
      long offset
      int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng
      int i1=gc.i1, i2=gc.i2
      int j1=gc.j1, j2=gc.j2
      int k1=gc.k1, k2=gc.k2
      int nbvar = bvars.size
      int1d lims, pack_order

    flds = get_bvar_fld_list(gd, bvars)
    pack_order = np.ones(3, dtype=np.intp)
    offset=0

    if side==0:
      if ax==0:   lims = np.array([i1,i1+ng, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j1,j1+ng, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k1,k1+ng])
    elif side==1:
      if ax==0:   lims = np.array([i2-ng+1,i2, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j2-ng+1,j2, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k2-ng+1,k2])

    for n in range(nbvar):
      pack(flds[n], buf, &offset, lims, pack_order, 1)

    return


  # -----------------------------------------------------------------------------

  cdef void unpack_grid_all(GridData gd, GridCoord *gc,  int1d bvars, real1d buf, int ax, int side):

    cdef:
      int n
      long offset
      int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng
      int i1=gc.i1, i2=gc.i2, j1=gc.j1, j2=gc.j2, k1=gc.k1, k2=gc.k2
      int nbvar = bvars.size
      int1d lims

    flds = get_bvar_fld_list(gd, bvars)
    offset=0

    # Now treat ordinary MPI boundaries.

    if side==0:
      if ax==0:   lims = np.array([i1-ng,i1-1, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j1-ng,j1-1, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k1-ng,k1-1])
    elif side==1:
      if ax==0:   lims = np.array([i2+1,i2+ng+1, 0,ny-1, 0,nz-1])
      elif ax==1: lims = np.array([0,nx-1, j2+1,j2+ng+1, 0,nz-1])
      elif ax==2: lims = np.array([0,nx-1, 0,ny-1, k2+1,k2+ng+1])

    for n in range(nbvar):
      unpack(flds[n], buf, &offset, lims)

    return
