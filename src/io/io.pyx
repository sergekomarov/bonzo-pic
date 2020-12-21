# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid
import os
import h5py

from read_config import read_param
from bnz.util cimport print_root, timediff
cimport bnz.problem.problem as problem
from output cimport write_history, write_grid, write_slice, write_particles
from restart cimport set_restart_grid, set_restart_particles

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef class BnzOutput:

  def __cinit__(str usr_dir):

    # Read parameters from user file.

    self.usr_dir = usr_dir

    self.hst_dt  = read_param("output",  "history_dt", 'f', usr_dir)
    self.grid_dt = read_param("output",  "grid_dt", 'f', usr_dir)
    self.slc_dt  = read_param("output",  "slice_dt", 'f', usr_dir)
    self.prt_dt  = read_param("output",  "particle_dt", 'f', usr_dir)
    self.rst_dt  = read_param("restart", "restart_dt", 'f', usr_dir)

    self.use_npy = read_param("output", "use_npy_arrays", 'i', usr_dir)
    IF MPI: self.use_npy=0

    self.restart = read_param("restart", "restart", 'i', usr_dir)

    slc_axis = read_param("output", "slice_axis", 's', usr_dir)
    if   slc_axis=='x': self.slc_axis=0
    elif slc_axis=='y': self.slc_axis=1
    elif slc_axis=='z': self.slc_axis=2

    self.slc_loc = read_param("output", "slice_cell_index", 'i', usr_dir)

    self.write_ghost = read_param("output", "write_ghost_cells", 'i', usr_dir)

    self.prt_stride = read_param("output", "particle_output_stride", 'i', usr_dir)

    # --------------------------------------------

    # set default number of history variables
    self.nhst = 38

    # set user history variables (function pointers and names)
    # set user particle selection function

    self.prt_sel_func=NULL
    self.hst_funcs_u=[]
    self.hst_names_u=[]

    problem.set_output_user(self)

    # number of user-defined history variables set by function pointers
    cdef int nhst_u = len(self.hst_funcs_u)
    # set total number of history variables
    self.nhst += nhst_u

    # set names of history variables

    # grid quantities
    hst_names = [ '[0]step', '[1]time', '[2]<E^2>/2','[3]<B^2>/2',
                  '[4]<Ex>', '[5]<Ey>', '[6]<Ez>',  '[7]rms(Ex)', '[8]rms(Ey)', '[9]rms(Ez)',
                  '[10]<Bx>','[11]<By>','[12]<Bz>', '[13]rms(Bx)','[14]rms(By)','[15]rms(Bz)',
                  '[16]<Jx>','[17]<Jy>','[18]<Jz>', '[19]rms(Jx)','[20]rms(Jy)','[21]rms(Jz)']
    # particle statistics
    hst_names += ['[22]rhoi', '[23]rhoe', '[24]<Eki>','[25]<Eke>',
                  '[26]<vxi>','[27]<vxe>','[28]<vyi>','[29]<vye>',
                  '[30]<vzi>','[31]<vze>','[32]rms(vxi)', '[33]rms(vxe)',
                  '[34]rms(vyi)', '[35]rms(vye)', '[36]rms(vzi)', '[37]rms(vze)']

    cdef int i
    for i in range(nhst_u):
      hst_names.append(self.hst_names_u[i])

    cdef int rank = 0
    IF MPI: rank = mpi.COMM_WORLD.Get_rank()

    if not self.restart and rank==0:

      # create output folder and history file

      out_dir = os.path.join(self.usr_dir,'out')
      if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

      rst_dir = os.path.join(self.usr_dir,'rst')
      if not os.path.isdir(rst_dir):
        os.mkdir(rst_dir)

      with open(os.path.join(out_dir,'hst.txt'), 'w') as f:
        # write names of history variables
        for name in hst_names:
          f.write('{:<14}'.format(name))
        f.write('\n')


  # -----------------------------------------------------------------------

  cdef void write_output(self, BnzGrid grid, BnzIntegr integr):

    cdef real dt,t1
    cdef timeval tstart, tstart_step, tstop

    # hack to allow writing initial conditions, when dt is not set yet
    dt = integr.dt if integr.time>0. else 1e-6

    if self.hst_dt != 0.:

      t1 = <int>(integr.time / self.hst_dt) * self.hst_dt
      if (integr.time-dt < t1) and (integr.time >= t1):

        print_root("writing history... ")
        gettimeofday(&tstart, NULL)

        write_history(self,grid,integr)

        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

    if self.grid_dt != 0.:

      t1 = <int>(integr.time / self.grid_dt) * self.grid_dt
      if (integr.time-dt < t1) and (integr.time >= t1):

        print_root("writing full grid... ")
        gettimeofday(&tstart, NULL)

        write_grid(self,grid,integr, 0)   # 0 for no restart

        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

    if self.slc_dt != 0.:

      t1 = <int>(integr.time / self.slc_dt) * self.slc_dt
      if (integr.time-dt < t1) and (integr.time >= t1):

        print_root("writing slice... ")
        gettimeofday(&tstart, NULL)

        write_slice(self,grid,integr)

        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

    if self.prt_dt != 0.:

      t1 = <int>(integr.time / self.prt_dt) * self.prt_dt
      if (integr.time-dt < t1) and (integr.time >= t1):

        print_root("writing particles... ")
        gettimeofday(&tstart, NULL)

        write_particles(self,grid,integr, 0)

        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))


  # ------------------------------------------------------------------

  cdef void write_restart(self, BnzGrid grid, BnzIntegr integr):

    cdef real t1
    cdef timeval tstart, tstart_step, tstop

    if self.rst_dt != 0.:

      t1 = <int>(integr.time / self.rst_dt) * self.rst_dt
      if integr.time-integr.dt < t1 and integr.time >= t1:

        print_root("writing restart files... ")
        gettimeofday(&tstart, NULL)

        write_grid(self,grid,integr, 1)
        write_particles(self,grid,integr, 1)

        gettimeofday(&tstop, NULL)
        print_root("%.1f ms\n", timediff(tstart,tstop))

  # ------------------------------------------------------------------

  cdef void set_restart(self, BnzGrid grid, BnzIntegr integr):

    print_root("restarting... ")

    set_restart_grid(self,grid,integr)
    set_restart_prt(self,grid,integr)
