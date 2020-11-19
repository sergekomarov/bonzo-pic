# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange, threadid

from libc.math cimport sqrt,floor,log,exp,sin,cos,pow,fabs,fmin,fmax

from libc.stdlib cimport calloc, free

from bnz.utils cimport calloc_2d_array, free_2d_array
from bnz.utils cimport mini, maxi, print_root, timediff

cimport bnz.bc.bc_grid as bc_grid



# ==============================================================================

cdef void deposit_curr(real4d J, BnzParticles prts, GridParams gp, double c) nogil:

  cdef:

    ints n, i,j,k, r,s
    ints ib1,jb1,kb1, ib2,jb2,kb2

    real x1,y1,z1, x2,y2,z2
    real u,v,w
    real xr,yr,zr
    real fx1,fy1,fz1, fx2,fy2,fz2

    real wx1[2]
    real wy1[2]
    real wz1[2]
    real wx2[2]
    real wy2[2]
    real wz2[2]

    double cinv2 = 1./c**2
    double ginv, qp

    ParticleData *pd = &(prts.data)


  # clear array of currents
  clear_curr(J, gp.Ntot)

  # deposit currents

  for n in range(prts.Np):

    # particle coordinates are local
    # active region starts at x,y,z=0

    x2 = pd.x[n]
    y2 = pd.y[n]
    z2 = pd.z[n]
    u,v,w = pd.u[n], pd.v[n], pd.w[n]

    # calculate particle coordinates at previous step (dt=1)

    ginv = 1. / sqrt(1. + cinv2 * (u**2 + v**2 + w**2))
    x1 = x2 - ginv * u
    y1 = y2 - ginv * v
    z1 = z2 - ginv * w

    # neighbouring cell indices (dl=1)

    # cell indices including ghost cells (+ng)
    ib1 = <ints>floor(x1) #dli[0] *
    jb1 = <ints>floor(y1) #dli[1] *
    kb1 = <ints>floor(z1) #dli[2] *

    ib2 = <ints>floor(x2) #dli[0] *
    jb2 = <ints>floor(y2) #dli[1] *
    kb2 = <ints>floor(z2) #dli[2] *

    # coordinates of the relay point (Umeda 2003)

    xr = fmin(fmin(ib1,ib2) + 1., fmax(fmax(ib1,ib2), 0.5*(x1+x2)))
    yr = fmin(fmin(jb1,jb2) + 1., fmax(fmax(jb1,jb2), 0.5*(y1+y2)))
    zr = fmin(fmin(kb1,kb2) + 1., fmax(fmax(kb1,kb2), 0.5*(z1+z2)))

    # charge fluxes

    # dt=1 for pure PIC, q/dt=q

    qp = pd.m[n] * prts.spc_props[pd.spc[n]].qm

    fx1 = qp * (xr-x1)
    fy1 = qp * (yr-y1)
    fz1 = qp * (zr-z1)

    fx2 = qp * (x2-xr)
    fy2 = qp * (y2-yr)
    fz2 = qp * (z2-zr)

    # weights for charge flux interpolation

    wx1[1] = 0.5 * (x1 + xr) - ib1
    IF D2D: wy1[1] = 0.5 * (y1 + yr) - jb1
    IF D3D: wz1[1] = 0.5 * (z1 + zr) - kb1
    wx1[0], wy1[0], wz1[0] = 1-wx1[1], 1-wy1[1], 1-wz1[1]

    wx2[1] = 0.5 * (xr + x2) - ib2
    IF D2D: wy2[1] = 0.5 * (yr + y2) - jb2
    IF D3D: wz2[1] = 0.5 * (zr + z2) - kb2
    wx2[0], wy2[0], wz2[0] = 1-wx2[1], 1-wy2[1], 1-wz2[1]

    ib1 += gp.ng
    IF D2D: jb1 += gp.ng
    IF D3D: kb1 += gp.ng

    # deposit current to starting-point

    IF D2D and D3D:

      for s in range(2):
        for r in range(2):
          J[0,kb1+s,jb1+r,ib1] += fx1 * wy1[r] * wz1[s]
          J[0,kb2+s,jb2+r,ib2] += fx2 * wy2[r] * wz2[s]

          J[1,kb1+s,jb1,ib1+r] += fy1 * wx1[r] * wz1[s]
          J[1,kb2+s,jb2,ib2+r] += fy2 * wx2[r] * wz2[s]

          J[2,kb1,jb1+s,ib1+r] += fz1 * wx1[r] * wy1[s]
          J[2,kb2,jb2+s,ib2+r] += fz2 * wx2[r] * wy2[s]

    ELIF D2D:

      for r in range(2):

        J[0,0,jb1+r,ib1] += fx1 * wy1[r]
        J[0,0,jb2+r,ib2] += fx2 * wy2[r]

        J[1,0,jb1,ib1+r] += fy1 * wx1[r]
        J[1,0,jb2,ib2+r] += fy2 * wx2[r]

      for s in range(2):
        for r in range(2):

          J[2,0,jb1+s,ib1+r] += fz1 * wx1[r] * wy1[s]
          J[2,0,jb2+s,ib2+r] += fz2 * wx2[r] * wy2[s]

    ELSE:

      J[0,0,0,ib1] += fx1
      J[0,0,0,ib2] += fx2

      for r in range(2):

        J[1,0,0,ib1+r] += fy1 * wx1[r]
        J[1,0,0,ib2+r] += fy2 * wx2[r]

        J[2,0,0,ib1+r] += fz1 * wx1[r]
        J[2,0,0,ib2+r] += fz2 * wx2[r]



# ============================================================

# Digital filter applied to deposited particle current.

cdef void filter_curr1_copy2(real4d J, real2d tmp1, real2d tmp2,
                             int nfilt, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,m,n
    ints Nx=Ntot[0], Ny=Ntot[1], Nz=Ntot[2]
    real Jm1,J0,Jp1,Jp2


  for m in range(nfilt):

    for n in range(3):

      for k in range(Nz):
        for j in range(Ny):

          tmp2[0,0] = J[n,k,j,0]

          for i in range(1,Nx-2,2):

            tmp1[0,0] = 0.25*J[n,k,j,i-1] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j,i+1]
            J[n,k,j,i-1] = tmp2[0,0]

            tmp2[0,0] = 0.25*J[n,k,j,i] + 0.5*J[n,k,j,i+1] + 0.25*J[n,k,j,i+2]
            J[n,k,j,i] = tmp1[0,0]

          i = i+2

          if i==Nx-2:

            tmp1[0,0] = 0.25*J[n,k,j,i-1] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j,i+1]
            J[n,k,j,i-1] = tmp2[0,0]
            J[n,k,j,i] = tmp1[0,0]

          elif i==Nx-1:

            J[n,k,j,i-1] = tmp2[0,0]

  #-----------------------------------------------

      IF D2D:

        for k in range(Nz):

          for i in range(Nx):

            tmp2[0,i] = J[n,k,0,i]

          for j in range(1,Ny-2,2):

            for i in range(Nx):

              Jm1,J0,Jp1,Jp2 = J[n,k,j-1,i], J[n,k,j,i], J[n,k,j+1,i], J[n,k,j+2,i]

              tmp1[0,i] = 0.25*Jm1 + 0.5*J0 + 0.25*Jp1
              J[n,k,j-1,i] = tmp2[0,i]

              tmp2[0,i] = 0.25*J0 + 0.5*Jp1 + 0.25*Jp2
              J[n,k,j,i] = tmp1[0,i]

          j = j+2

          if j==Ny-2:

            for i in range(Nx):

              tmp1[0,i] = 0.25*J[n,k,j-1,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j+1,i]
              J[n,k,j-1,i] = tmp2[0,i]
              J[n,k,j,i] = tmp1[0,i]

          elif j==Ny-1:

            for i in range(Nx):
              J[n,k,j-1,i] = tmp2[0,i]

  #----------------------------------------------

      IF D3D:

        for j in range(Ny):
          for i in range(Nx):

            tmp2[j,i] = J[n,0,j,i]

        for k in range(1,Nz-2,2):

          for j in range(Ny):
            for i in range(Nx):

              Jm1,J0,Jp1,Jp2 = J[n,k-1,j,i], J[n,k,j,i], J[n,k+1,j,i], J[n,k+2,j,i]

              tmp1[j,i] = 0.25*Jm1 + 0.5*J0 + 0.25*Jp1
              J[n,k-1,j,i] = tmp2[j,i]

              tmp2[j,i] = 0.25*J0 + 0.5*Jp1 + 0.25*Jp2
              J[n,k,j,i] = tmp1[j,i]

        k = k+2

        if k==Nz-2:

          for j in range(Ny):
            for i in range(Nx):

              tmp1[j,i] = 0.25*J[n,k-1,j,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k+1,j,i]
              J[n,k-1,j,i] = tmp2[j,i]
              J[n,k,j,i] = tmp1[j,i]

        elif k==Nz-1:

          for j in range(Ny):
            for i in range(Nx):
              J[n,k-1,j,i] = tmp2[j,i]


# ============================================================

# Digital filter applied to deposited particle current.

cdef void filter_curr1_copy3(real4d J, real4d Jtmp, int nfilt, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,m,n
    ints Nx=Ntot[0], Ny=Ntot[1], Nz=Ntot[2]


  for m in range(nfilt):

    for n in range(3):
      for k in range(Nz):
        for j in range(Ny):
          for i in range(Nx):
            Jtmp[n,k,j,i] = J[n,k,j,i]

    for n in range(3):

      IF D3D:

        for j in range(Ny):
          for i in range(Nx):
            for k in range(1,Nz-1):

              J[n,k,j,i] = 0.25*J[n,k-1,j,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k+1,j,i]

      IF D2D:

        for k in range(Nz):
          for i in range(Nx):
            for j in range(1,Ny-1):

              J[n,k,j,i] = 0.25*J[n,k,j-1,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j+1,i]

      for k in range(Nz):
        for j in range(Ny):
          for i in range(1,Nx-1):

            J[n,k,j,i] = 0.25*J[n,k,j,i-1] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j,i+1]




# ============================================================

# Digital filter applied to deposited particle current.

cdef void filter_curr1(real4d J, int nfilt, ints Ntot[3]) nogil:

  cdef:
    ints i,j,k,m,n
    real tmp1, tmp2
    ints Nx=Ntot[0], Ny=Ntot[1], Nz=Ntot[2]


  for m in range(nfilt):
    for n in range(3):

      IF D3D:

        for j in range(Ny):
          for i in range(Nx):

            tmp2 = J[n,0,j,i]

            for k in range(1,Nz-2,2):

              tmp1 = 0.25*J[n,k-1,j,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k+1,j,i]
              J[n,k-1,j,i] = tmp2

              tmp2 = 0.25*J[n,k,j,i] + 0.5*J[n,k+1,j,i] + 0.25*J[n,k+2,j,i]
              J[n,k,j,i] = tmp1

            k = k+2

            if k==Nz-2:
              tmp1 = 0.25*J[n,k-1,j,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k+1,j,i]
              J[n,k-1,j,i] = tmp2
              J[n,k,j,i] = tmp1

            elif k==Nz-1:
              J[n,k-1,j,i] = tmp2

      IF D2D:

        for k in range(Nz):
          for i in range(Nx):

            tmp2 = J[n,k,0,i]

            for j in range(1,Ny-2,2):

              tmp1 = 0.25*J[n,k,j-1,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j+1,i]
              J[n,k,j-1,i] = tmp2

              tmp2 = 0.25*J[n,k,j,i] + 0.5*J[n,k,j+1,i] + 0.25*J[n,k,j+2,i]
              J[n,k,j,i] = tmp1

            j = j+2

            if j==Ny-2:
              tmp1 = 0.25*J[n,k,j-1,i] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j+1,i]
              J[n,k,j-1,i] = tmp2
              J[n,k,j,i] = tmp1

            elif j==Ny-1:
              J[n,k,j-1,i] = tmp2


      for k in range(Nz):
        for j in range(Ny):

          tmp2 = J[n,k,j,0]

          for i in range(1,Nx-2,2):

            tmp1 = 0.25*J[n,k,j,i-1] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j,i+1]
            J[n,k,j,i-1] = tmp2

            tmp2 = 0.25*J[n,k,j,i] + 0.5*J[n,k,j,i+1] + 0.25*J[n,k,j,i+2]
            J[n,k,j,i] = tmp1

          i = i+2

          if i==Nx-2:
            tmp1 = 0.25*J[n,k,j,i-1] + 0.5*J[n,k,j,i] + 0.25*J[n,k,j,i+1]
            J[n,k,j,i-1] = tmp2
            J[n,k,j,i] = tmp1

          elif i==Nx-1:
            J[n,k,j,i-1] = tmp2

#-----------------------------------------------------------

cdef void filter_curr(real4d J, BnzSim sim, GridParams gp, int Nfilt):

  cdef:
    ints n
    ints N = Nfilt / gp.ng
    ints nfilt = Nfilt % gp.ng

  bc_grid.apply_bc_grid(sim, np.array([JX,JY,JZ]))

  for n in range(N):
    filter_curr1(J, gp.ng, gp.Ntot)
    if n != N-1:
      bc_grid.apply_bc_grid(sim, np.array([JX,JY,JZ]))

  filter_curr1(J, nfilt, gp.Ntot)


# ============================================================

cdef void clear_curr(real4d J, ints Ntot[3]) nogil:

  cdef ints i,j,k,n

  for n in range(3):
    for k in range(Ntot[2]):
      for j in range(Ntot[1]):
        for i in range(Ntot[0]):
          J[n,k,j,i] = 0.
