# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdio cimport printf

from bnz.utils cimport print_root, timediff

cimport diagnostics_pic as diagnostics
cimport bnz.bc.bc_grid as bc_grid
cimport bnz.bc.bc_pic_exch as bc_pic_exch
cimport bnz.bc.bc_prt as bc_prt

cimport field
cimport deposit
cimport move

cimport bnz.output as output


# ==============================================================================

# PIC integrator.

cdef void integrate(BnzSim sim):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data
    BnzParticles prts = sim.prts
    BnzPhysics phys = sim.phys
    BnzMethod method = sim.method

  # C structs to measure timings
  cdef timeval tstart, tstart_step, tstop

  cdef ints rank=0
  IF MPI: rank = mpi.COMM_WORLD.Get_rank()

  cdef:
    ints lims_b[6]
    ints lims_e[6]

  lims_b[:] = [gp.i1-1,gp.i2+1, gp.j1-1,gp.j2+1, gp.k1-1,gp.k2+1]
  lims_e[:] = [gp.i1,gp.i2, gp.j1,gp.j2, gp.k1,gp.k2]
  IF not D2D:
    lims_b[2], lims_b[3] = 0, 0
    lims_e[2], lims_e[3] = 0, 0
  IF not D3D:
    lims_b[4], lims_b[5] = 0, 0
    lims_e[4], lims_e[5] = 0, 0


  #=============================
  # start main integration loop
  #=============================

  while sim.t < sim.tmax:

    print_root(rank,
        "\n==========step %i, time = %.5f==========\n\n",
        sim.step+1, sim.t)

    gettimeofday(&tstart_step, NULL)


    #---------------------------------------------------------------------------

    print_root(rank, "advance B-field by a half-step... ")
    gettimeofday(&tstart, NULL)

    field.advance_b_field(gd.B, gd.E, lims_b, 0.5, phys.c)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "move particles... ")
    gettimeofday(&tstart, NULL)

    move.move_particles(prts, grid, phys.c)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "advance B-field by a half-step... ")
    gettimeofday(&tstart, NULL)

    field.advance_b_field(gd.B, gd.E, lims_b, 0.5, phys.c)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "deposit currents... ")
    gettimeofday(&tstart, NULL)

    deposit.deposit_curr(gd.J, prts, gp, phys.c)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "exchange boundary currents... ")
    gettimeofday(&tstart, NULL)

    bc_pic_exch.apply_bc_exch(sim)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "filter currents... ")
    gettimeofday(&tstart, NULL)

    deposit.filter_curr(gd.J, sim, gp, method.Nfilt) # need sim to apply BCs

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "advance E-field by a full time-step... ")
    gettimeofday(&tstart, NULL)

    field.advance_e_field(gd.E, gd.B, gd.J, lims_e, 1., phys.c)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "appply particle BC... ")
    gettimeofday(&tstart, NULL)

    bc_prt.apply_bc_prt(sim)

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    print_root(rank, "appply field BC... ")
    gettimeofday(&tstart, NULL)

    bc_grid.apply_bc_grid(sim, np.array([EX,EY,EZ,BX,BY,BZ]))

    gettimeofday(&tstop, NULL)
    print_root(rank, "%.1f ms\n", timediff(tstart,tstop))

    #---------------------------------------------------------------------------

    # reorder_particles()

    #---------------------------------------------------------------------------

    # update time and step
    sim.t = sim.t + sim.dt
    sim.step = sim.step + 1

    #---------------------------------------------------------------------------

    # write output and restart files
    output.write_output(sim)
    output.write_restart(sim)

    #---------------------------------------------------------------------------

    # print mean energy densities
    gettimeofday(&tstart, NULL)

    diagnostics.print_nrg(sim)

    gettimeofday(&tstop, NULL)
    print_root(rank, "mean energy densities calculated in ")
    print_root(rank, "%.1f ms\n\n", timediff(tstart,tstop))

    # print timings for complete timestep
    gettimeofday(&tstop, NULL)
    print_root(rank, "step %i completed in %.1f ms\n\n",
            sim.step, timediff(tstart_step,tstop))
