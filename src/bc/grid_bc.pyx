# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np
import sys

from util cimport print_root
from problem cimport set_grid_bc_user
from grid_bc_funcs cimport *
from read_config import read_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64
IF MPI:
  IF SPREC:
    mpi_real = mpi.FLOAT
  ELSE:
    mpi_real = mpi.DOUBLE


cdef class GridBc:

  def __cinit__(self, GridCoord gc, str usr_dir):

    cdef int i,k

    for i in range(3):
      for k in range(2):
        self.grid_bc_funcs[i][k] = NULL

    # Set boundary condition flags.

    self.bc_flags[0][0] = read_param("physics", "bc_x1", 'i',usr_dir)
    self.bc_flags[0][1] = read_param("physics", "bc_x2", 'i',usr_dir)
    self.bc_flags[1][0] = read_param("physics", "bc_y1", 'i',usr_dir)
    self.bc_flags[1][1] = read_param("physics", "bc_y2", 'i',usr_dir)
    self.bc_flags[2][0] = read_param("physics", "bc_z1", 'i',usr_dir)
    self.bc_flags[2][1] = read_param("physics", "bc_z2", 'i',usr_dir)

    # Set BC function pointers.

    cdef:
      int sx=gc.size[0], sy=gc.size[1], sz=gc.size[2]
      int px=gc.pos[0], py=gc.pos[1], pz=gc.pos[2]

    # left x boundary

    if self.bc_flags[0][0]==0:
      self.grid_bc_funcs[0][0] = x1_grid_bc_periodic

      IF MPI:
        if px==0 and sx>1:
          gc.nbr_ranks[0][0] = gc.ranks[sx-1][py][pz]

    if self.bc_flags[0][0]==1:
      self.grid_bc_funcs[0][0] = x1_grid_bc_outflow

    if self.bc_flags[0][0]==2:
      self.grid_bc_funcs[0][0] = x1_bc_pic_conduct

    # right x boundary

    if self.bc_flags[0][1]==0:
      self.grid_bc_funcs[0][1] = x2_grid_bc_periodic

      IF MPI:
        if px==sx-1 and sx>1:
          gc.nbr_ranks[0][1] = gc.ranks[0][py][pz]

    if self.bc_flags[0][1]==1:
      self.grid_bc_funcs[0][1] = x2_grid_bc_outflow

    if self.bc_flags[0][1]==2:
      self.grid_bc_funcs[0][1] = x2_bc_pic_conduct

    # left y boundary

    if self.bc_flags[1][0]==0:
      self.grid_bc_funcs[1][0] = y1_grid_bc_periodic

      IF MPI:
        if py==0 and sy>1:
          gc.nbr_ranks[1][0] = gc.ranks[px][sy-1][pz]

    if self.bc_flags[1][0]==1:
      self.grid_bc_funcs[1][0] = y1_grid_bc_outflow

    if self.bc_flags[1][0]==2:
      self.grid_bc_funcs[1][0] = y1_bc_pic_conduct

    # right y boundary

    if self.bc_flags[1][1]==0:
      self.grid_bc_funcs[1][1] = y2_grid_bc_periodic

      IF MPI:
        if py==sy-1 and sy>1:
          gc.nbr_ranks[1][1] = gc.ranks[px][0][pz]

    if self.bc_flags[1][1]==1:
      self.grid_bc_funcs[1][1] = y2_grid_bc_outflow

    if self.bc_flags[1][1]==2:
      self.grid_bc_funcs[1][1] = y2_bc_pic_conduct

    # left z boundary

    if self.bc_flags[2][0]==0:
      self.grid_bc_funcs[2][0] = z1_grid_bc_periodic

      IF MPI:
        if pz==0 and sz>1:
          gc.nbr_ranks[2][0] = gc.ranks[px][py][sz-1]

    if self.bc_flags[2][0]==1:
      self.grid_bc_funcs[2][0] = z1_grid_bc_outflow

    if self.bc_flags[2][0]==2:
      self.grid_bc_funcs[2][0] = z1_bc_pic_conduct

    # right z boundary

    if self.bc_flags[2][1]==0:
      self.grid_bc_funcs[2][1] = z2_grid_bc_periodic

      IF MPI:
        if pz==sz-1 and sz>1:
          gc.nbr_ranks[2][1] = gc.ranks[px][py][0]

    if self.bc_flags[2][1]==1:
      self.grid_bc_funcs[2][1] = z2_grid_bc_outflow

    if self.bc_flags[2][1]==2:
      self.grid_bc_funcs[2][1] = z2_bc_pic_conduct


    # Set remaining user-defined BCs.

    set_grid_bc_user(self)

    # Check if all BCs have been set.

    cdef int i,k

    for i in range(3):
      for k in range(2):
        if self.grid_bc_funcs[i][k] == NULL:
          if k==0:
            print_root('\nWARNING: Left boundary condition in %i direction is not set\n')
          else:
            print_root('\nWARNING: Right boundary condition in %i direction is not set\n')


    IF MPI:

      # Allocate BC buffers for MPI.

      cdef:
        int n, ndim=1
        long bufsize
        long Nxyz = IMAX(IMAX(gc.Ntot[0],gc.Ntot[1]), gc.Ntot[2])

      IF D2D: ndim += 1
      IF D3D: ndim += 1

      n = NVAR * (gc.ng+1)

      if ndim=3:
        bufsize = (Nxyz+1)**2 * n
      elif ndim==2:
        bufsize = (Nxyz+1) * n
      else:
        bufsize = n

      self.sendbuf = np.zeros((2,bufsize), dtype=np_real)
      self.recvbuf = np.zeros((2,bufsize), dtype=np_real)
      self.recvbuf_size = bufsize
      self.sendbuf_size = bufsize


  #-------------------------------------------------------------------

  def __dealloc__(self):

    cdef int i,k

    for i in range(3):
      for k in range(2):
        self.grid_bc_funcs[i][k] = NULL


  # ------------------------------------------------------------------

  cdef void apply_grid_bc(self, GridData gd, GridCoord *gc,
                          BnzIntegr integr, int1d bvars):

    IF MPI:

      cdef int nx=gc.Ntot[0], ny=gc.Ntot[1], nz=gc.Ntot[2], ng=gc.ng

      if bvars is None:
        # if bvars not specified, apply BC to all variables
        bvars = np.arange(NVAR, dype=np.intp)

      cdef int nbvar = bvars.size

      cdef:

        mpi.Comm comm = mpi.COMM_WORLD
        int done
        mpi.Request send_req1, send_req2, recv_req1, recv_req2

        int cnt1,cnt2

        real2d sendbuf = self.sendbuf
        real2d recvbuf = self.recvbuf
        int **nbr_ranks = gc.nbr_ranks

      cdef:
        int rtagl=1, rtagr=0
        int stagl=0, stagr=1

      # ------- data exchange in x-direction --------------


      cnt1 = ny * nz *  ng    * nbvar
      cnt2 = ny * nz * (ng+1) * nbvar

      if nbr_ranks[0][0] > -1 and nbr_ranks[0][1] > -1:

        recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[0][0], tag=rtagl)
        recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[0][1], tag=rtagr)

        pack_grid_all(gd,gc, bvars, sendbuf[0,:], XAX, 0)
        send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[0][0], tag=stagl)

        pack_grid_all(gd,gc, bvars, sendbuf[1,:], XAX, 1)
        send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[0][1], tag=stagr)

        mpi.Request.Waitall([send_req1,send_req2])

        done = mpi.Request.Waitany([recv_req1,recv_req2])
        if done==0:   unpack_grid_all(gd,gc, bvars, recvbuf[0,:], XAX,0)
        elif done==1: unpack_grid_all(gd,gc, bvars, recvbuf[1,:], XAX,1)

        done = mpi.Request.Waitany([recv_req1,recv_req2])
        if done==0:   unpack_grid_all(gd,gc, bvars, recvbuf[0,:], XAX,0)
        elif done==1: unpack_grid_all(gd,gc, bvars, recvbuf[1,:], XAX,1)

      elif nbr_ranks[0][0] == -1 and nbr_ranks[0][1] > -1:

        recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[0][1], tag=rtagr)

        pack_grid_all(gd,gc, bvars, sendbuf[1,:], XAX, 1)
        send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[0][1], tag=stagr)

        self.grid_bc_funcs[0][0](sim, bvars)

        send_req2.Wait()
        recv_req2.Wait()
        unpack_grid_all(gd,gc, bvars, recvbuf[1,:], XAX,1)

      elif nbr_ranks[0][0] > -1 and nbr_ranks[0][1] == -1:

        recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[0][0], tag=rtagl)

        pack_grid_all(gd,gc, bvars, sendbuf[0,:], XAX, 0)
        send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[0][0], tag=stagl)

        self.grid_bc_funcs[0][1](sim, bvars)

        send_req1.Wait()
        recv_req1.Wait()
        unpack_grid_all(gd,gc, bvars, recvbuf[0,:], XAX,0)

      else:

        self.grid_bc_funcs[0][0](sim, bvars)
        self.grid_bc_funcs[0][1](sim, bvars)


      # ------- data exchange in y-direction --------------

      IF D2D:

        cnt1 = nx * nz *  ng    * nbvar
        cnt2 = nx * nz * (ng+1) * nbvar

        if nbr_ranks[1][0] > -1 and nbr_ranks[1][1] > -1:

          recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[1][0], tag=rtagl)
          recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[1][1], tag=rtagr)

          pack_grid_all(gd,gc, bvars, sendbuf[0,:], YAX, 0)
          send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[1][0], tag=stagl)

          pack_grid_all(gd,gc, bvars, sendbuf[1,:], YAX, 1)
          send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[1][1], tag=stagr)

          mpi.Request.Waitall([send_req1,send_req2])

          done = mpi.Request.Waitany([recv_req1,recv_req2])
          if done==0:   unpack_grid_all(gd,gc, bvars, recvbuf[0,:], YAX,0)
          elif done==1: unpack_grid_all(gd,gc, bvars, recvbuf[1,:], YAX,1)

          done = mpi.Request.Waitany([recv_req1,recv_req2])
          if done==0:   unpack_grid_all(gd,gc, bvars, recvbuf[0,:], YAX,0)
          elif done==1: unpack_grid_all(gd,gc, bvars, recvbuf[1,:], YAX,1)

        elif nbr_ranks[1][0] == -1 and nbr_ranks[1][1] > -1:

          recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[1][1], tag=rtagr)

          pack_grid_all(gd,gc, bvars, sendbuf[1,:], YAX, 1)
          send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[1][1], tag=stagr)

          self.grid_bc_funcs[1][0](sim, bvars)

          send_req2.Wait()
          recv_req2.Wait()
          unpack_grid_all(gd,gc, bvars, recvbuf[1,:], YAX,1)

        elif nbr_ranks[1][0] > -1 and nbr_ranks[1][1] == -1:

          recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[1][0], tag=rtagl)

          pack_grid_all(gd,gc, bvars, sendbuf[0,:], YAX, 0)
          send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[1][0], tag=stagl)

          self.grid_bc_funcs[1][1](sim, bvars)

          send_req1.Wait()
          recv_req1.Wait()
          unpack_grid_all(gd,gc, bvars, recvbuf[0,:], YAX,0)

        else:

          self.grid_bc_funcs[1][0](sim, bvars)
          self.grid_bc_funcs[1][1](sim, bvars)

      # ------- data exchange in z-direction --------------

      IF D3D:

        cnt1 = nx * ny *  ng    * nbvar
        cnt2 = nx * ny * (ng+1) * nbvar

        if nbr_ranks[2][0] > -1 and nbr_ranks[2][1] > -1:

          recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[2][0], tag=rtagl)
          recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[2][1], tag=rtagr)

          pack_grid_all(gd,gc, bvars, sendbuf[0,:], ZAX, 0)
          send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[2][0], tag=stagl)
          pack_grid_all(gd,gc, bvars, sendbuf[1,:], ZAX, 1)
          send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[2][1], tag=stagr)

          mpi.Request.Waitall([send_req1,send_req2])

          done = mpi.Request.Waitany([recv_req1,recv_req2])
          if done==0:   unpack_grid_all(gd,gc, bvars, recvbuf[0,:], ZAX,0)
          elif done==1: unpack_grid_all(gd,gc, bvars, recvbuf[1,:], ZAX,1)

          done = mpi.Request.Waitany([recv_req1,recv_req2])
          if done==0:   unpack_grid_all(gd,gc, bvars, recvbuf[0,:], ZAX,0)
          elif done==1: unpack_grid_all(gd,gc, bvars, recvbuf[1,:], ZAX,1)

        elif nbr_ranks[2][0] == -1 and nbr_ranks[2][1] > -1:

          recv_req2 = comm.Irecv([recvbuf[1,:], cnt2, mpi_real], nbr_ranks[2][1], tag=rtagr)

          pack_grid_all(gd,gc, bvars, sendbuf[1,:], ZAX, 1)
          send_req2 = comm.Isend([sendbuf[1,:], cnt1, mpi_real], nbr_ranks[2][1], tag=stagr)

          self.grid_bc_funcs[2][0](sim, bvars)

          send_req2.Wait()
          recv_req2.Wait()
          unpack_grid_all(gd,gc, bvars, recvbuf[1,:], ZAX,1)

        elif nbr_ranks[2][0] > -1 and nbr_ranks[2][1] == -1:

          recv_req1 = comm.Irecv([recvbuf[0,:], cnt1, mpi_real], nbr_ranks[2][0], tag=rtagl)

          pack_grid_all(gd,gc, bvars, sendbuf[0,:], ZAX, 0)
          send_req1 = comm.Isend([sendbuf[0,:], cnt2, mpi_real], nbr_ranks[2][0], tag=stagl)

          self.grid_bc_funcs[2][1](sim, bvars)

          send_req1.Wait()
          recv_req1.Wait()
          unpack_grid_all(gd,gc, bvars, recvbuf[0,:], ZAX,0)

        else:

          self.grid_bc_funcs[2][0](sim, bvars)
          self.grid_bc_funcs[2][1](sim, bvars)

    ELSE:

      self.grid_bc_funcs[0][0](sim, bvars)
      self.grid_bc_funcs[0][1](sim, bvars)
      IF D2D:
        self.grid_bc_funcs[1][0](sim, bvars)
        self.grid_bc_funcs[1][1](sim, bvars)
      IF D3D:
        self.grid_bc_funcs[2][0](sim, bvars)
        self.grid_bc_funcs[2][1](sim, bvars)
