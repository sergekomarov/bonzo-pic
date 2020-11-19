# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
import sys, os

from libc.math cimport sqrt, fmax
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, calloc, free

import ConfigParser

from bnz.utils cimport maxi,mini, print_root

cimport bnz.read_config as read_config
cimport bnz.output as output
cimport bnz.restart as restart
IF MPI:
  cimport bnz.init_mpi_blocks as init_mpi_blocks
cimport bnz.problem.problem as problem
cimport bnz.bc.bc_grid as bc_grid
cimport bnz.bc.bc_prt as bc_prt

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


# ====================================================

# Initialize everything, prepare for integration.

cdef void init(BnzSim sim, bytes usr_dir):

  cdef ints rank = 0
  IF MPI: rank = mpi.COMM_WORLD.Get_rank()

  # Try to obtain parameters from the configuration file.
  print_root(rank, 'read the input file...\n')
  sim.output.usr_dir = usr_dir
  read_config.set_params(sim)

  print_root(rank, 'initialize grids...\n')
  init_grids(sim.grid, sim.phys, sim.method)

  print_root(rank, 'initialize particles...\n')
  init_particles(sim.prts, sim.phys, sim.method, sim.grid.params.Nact)

  if not sim.output.restart:
    print_root(rank, 'set the problem...\n')
    problem.set_problem(sim)
    sim.t = 0.
    sim.step = 0
  else:
    print_root(rank, 'restart...\n')
    restart.set_restart(sim)

  # print 'rank =', rank, '|', dom.blocks.x1nbr_id, dom.blocks.x2nbr_id, dom.blocks.y1nbr_id, dom.blocks.y2nbr_id

  print_root(rank, 'set grid boundary condition pointers...\n')
  bc_grid.set_bc_grid_ptrs(sim.bc, sim.grid.params)
  # if rank==0: np.save('blocks2.npy', np.asarray(dom.blocks.ids))

  # print 'rank =', rank, '|', dom.blocks.x1nbr_id, dom.blocks.x2nbr_id, dom.blocks.y1nbr_id, dom.blocks.y2nbr_id

  print_root(rank, 'set particle boundary condition pointers...\n')
  bc_prt.set_bc_prt_ptrs(sim.bc, sim.grid.params)

  IF MPI:
    print_root(rank, 'allocate BC buffers (MPI)...\n')
    init_mpi_blocks.init_bc_buffers(sim)

  print_root(rank, 'apply grid BC...\n')
  bc_grid.apply_bc_grid(sim, np.array(EX,EY,EZ,BX,BY,BZ))

  print_root(rank, 'initialize output...\n')
  output.init_output(sim)

  if not sim.output.restart:
    print_root(rank, 'write initial conditions...\n')
    output.write_output(sim)



#=============================================================

# Initialize grids and MHD data structures.

cdef void init_grids(BnzGrid grid, BnzPhysics phys, BnzMethod method):

  cdef:
    GridData gd = grid.data
    GridParams gp = grid.params

  IF not D2D:
    gp.Nact[1] = 1
  IF not D3D:
    gp.Nact[2] = 1

  # cell size
  gp.dl[0] = 1.
  gp.dl[1] = 1.
  gp.dl[2] = 1.

  gp.dli[0] = 1.
  gp.dli[1] = 1.
  gp.dli[2] = 1.

  gp.ng = mini(4, maxi(1, method.Nfilt))

  # decompose domain into MPI blocks
  IF MPI: init_mpi_blocks.init_blocks(gp)
    # after this dom.N and dom.L are LOCAL

  # box size including ghost cells
  gp.Ntot[0] = gp.Nact[0] + 2*gp.ng + 1
  # need 1 more ghost cell on the right for PIC
  IF D2D: gp.Ntot[1] = gp.Nact[1] + 2*gp.ng + 1
  ELSE: gp.Ntot[1] = 1
  IF D3D: gp.Ntot[2] = gp.Nact[2] + 2*gp.ng + 1
  ELSE: gp.Ntot[2] = 1

  IF not MPI:
    for k in range(3):
      gp.Nact_glob[k] = gp.Nact[k]
      gp.Ntot_glob[k] = gp.Ntot[k]
      gp.Lglob[k] = gp.L[k]

  # min and max local indices of active cells

  gp.i1, gp.i2 = gp.ng, gp.Nact[0] + gp.ng - 1

  IF D2D: gp.j1, gp.j2 = gp.ng, gp.Nact[1] + gp.ng - 1
  ELSE: gp.j1, gp.j2 = 0,0

  IF D3D: gp.k1, gp.k2 = gp.ng, gp.Nact[2] + gp.ng - 1
  ELSE: gp.k1, gp.k2 = 0,0


  gd.E = np.zeros((3,gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)
  gd.B = np.zeros((3,gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)
  gd.J = np.zeros((3,gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)

  IF MPI:
    grid.scr.Jtmp = np.zeros((3,gp.Ntot[2], gp.Ntot[1], gp.Ntot[0]), dtype=np_real)

  # initialize random number generator with different seeds
  IF MPI:
    srand(gp.rank)
  ELSE:
    np.random.seed()
    srand(np.random.randint(RAND_MAX))



# ==============================================================================

# Initialize particle data structures and set kernel function pointers.

cdef void init_particles(BnzParticles prts, BnzPhysics phys,
                         BnzMethod method, ints Nact[3]):


  # set total number of particles
  # need even total number of particles per cell for pure PIC (electrons + ions)

  IF D2D and D3D:
    prts.ppc = (<ints>(prts.ppc**(1./3)))**3
    if prts.ppc % 2 != 0: prts.ppc += 1
    prts.Np = prts.ppc * Nact[0] * Nact[1] * Nact[2]

  ELIF D2D:
    prts.ppc = (<ints>sqrt(prts.ppc))**2
    if prts.ppc % 2 != 0: prts.ppc += 1
    prts.Np = prts.ppc * Nact[0] * Nact[1]

  ELSE:
    if prts.ppc % 2 != 0: prts.ppc += 1
    prts.Np = prts.ppc * Nact[0]

  prts.Nprop = 10

  # allocate particle array

  prts.Npmax = prts.Np      # only when there is no injection of particles !!!
  IF MPI: prts.Npmax = <ints>(1.3*prts.Npmax)

  cdef ParticleData *pd = &(prts.data)

  pd.x = <real *>calloc(prts.Npmax, sizeof(real))
  pd.y = <real *>calloc(prts.Npmax, sizeof(real))
  pd.z = <real *>calloc(prts.Npmax, sizeof(real))

  pd.u = <real *>calloc(prts.Npmax, sizeof(real))
  pd.v = <real *>calloc(prts.Npmax, sizeof(real))
  pd.w = <real *>calloc(prts.Npmax, sizeof(real))
  pd.g = <real *>calloc(prts.Npmax, sizeof(real))

  pd.m = <real *>calloc(prts.Npmax, sizeof(real))
  pd.spc = <ints *>calloc(prts.Npmax, sizeof(ints))
  pd.id = <ints *>calloc(prts.Npmax, sizeof(ints))

  # allocate array that stores properties of different particle species
  prts.Ns=2    # 0: ions; 1: electrons by default
  prts.spc_props = <SpcProps *>calloc(prts.Ns, sizeof(SpcProps))

  phys.c = method.cour
  phys.me = phys.c / (phys.c_ompe * prts.ppc)
  prts.spc_props[0].qm =  1./phys.mime
  prts.spc_props[1].qm = -1.
  prts.spc_props[0].Np=  <ints>(0.5*prts.Np)
  prts.spc_props[1].Np=  <ints>(0.5*prts.Np)
