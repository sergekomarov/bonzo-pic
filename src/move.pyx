# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange, threadid

from libc.math cimport sqrt,floor,log,exp,sin,cos,pow,fabs,fmin,fmax

# from openmp cimport omp_set_lock, omp_unset_lock

from bnz.utils cimport mini, maxi, print_root, timediff



# ==================================================================

# Vay particle pusher (use for relativistic particles).

cdef inline void move_single_particle_vay(real *xp, real *yp, real *zp,
                        real *up, real *vp, real *wp, real *gp, double qm,
                        real ex, real ey, real ez,
                        real bx, real by, real bz,
                        double c, double cinv) nogil:

  cdef:
    real vx0,vy0,vz0, u1,v1,w1,g1
    real taux,tauy,tauz, tau2, tx,ty,tz, u1t

    real qmh, qmh_c, cinv2
    real ginv0, gfinv

    real sig, us, s

    real x,y,z, u,v,w, g
    real xf,yf,zf, uf,vf,wf, gf

  x,y,z = xp[0],yp[0],zp[0]
  u,v,w,g = up[0],vp[0],wp[0],gp[0]

  qmh = 0.5*qm
  qmh_c = qmh*cinv

  cinv2 = cinv**2

  # dt=1

  ex = qmh*ex
  ey = qmh*ey
  ez = qmh*ez

  taux = qmh_c * bx
  tauy = qmh_c * by
  tauz = qmh_c * bz

  tau2 = taux**2 + tauy**2 + tauz**2

  ginv0 = 1. / sqrt(1 + cinv2 * (u**2 + v**2 + w**2))
  vx0 = ginv0 * u
  vy0 = ginv0 * v
  vz0 = ginv0 * w

  u1 = u + 2*ex + vx0 * tauz - vz0 * tauy
  v1 = v + 2*ey + vz0 * taux - vx0 * tauz
  w1 = w + 2*ez + vx0 * tauy - vy0 * taux

  g1 = sqrt(1. + cinv2 * (u1**2 + v1**2 + w1**2))
  sig = g1**2 - tau2
  us = cinv * (u1 * taux + v1 * tauy + w1 * tauz)

  gf = sqrt(0.5*(sig + sqrt(sig**2 + 4*(tau2 + us**2))))
  gfinv = 1./gf

  tx = gfinv * taux
  ty = gfinv * tauy
  tz = gfinv * tauz

  s = 1 / (1 + tau2 * gfinv**2)
  u1t = c*gfinv * us

  uf = s * (u1 + u1t * tx + v1 * tz - w1 * ty)
  vf = s * (v1 + u1t * ty + w1 * tx - u1 * tz)
  wf = s * (w1 + u1t * tz + u1 * ty - v1 * tx)

  xf = x + gfinv * uf
  IF D2D: yf = y + gfinv * vf
  ELSE: yf=y
  IF D3D: zf = z + gfinv * wf
  ELSE: zf=z

  xp[0],yp[0],zp[0] = xf,yf,zf
  up[0],vp[0],wp[0],gp[0] = uf,vf,wf,gf



# ==========================================================================

# Interpolate electric and magnetic field at (x,y,z).

cdef inline void interp_fields_to_loc(real *ex, real *ey, real *ez,
                                      real *bx, real *by, real *bz,
                                      real x, real y, real z,
                                      real4d E, real4d B, ints ng) nogil:

  cdef:
    ints ib, jb=0, kb=0,  iu, ju=0, ku=0
    ints ibh,jbh=0,kbh=0, iuh,juh=0,kuh=0
    real wxb, wyb=0, wzb=0,  wxu, wyu=0, wzu=0
    real wxbh,wybh=0,wzbh=0, wxuh,wyuh=0,wzuh=0


  ib = <ints>floor(x) + ng         # dli[0] *
  IF D2D: jb = <ints>floor(y) + ng         # dli[1] *
  IF D3D: kb = <ints>floor(z) + ng         # dli[2] *

  ibh = <ints>floor(x-0.5) + ng    # dli[0] *
  IF D2D: jbh = <ints>floor(y-0.5) + ng    # dli[1] *
  IF D3D: kbh = <ints>floor(z-0.5) + ng    # dli[2] *

  iu, ju, ku = ib+1, jb+1, kb+1
  wxu, wyu, wzu = x-ib, y-jb, z-kb
  wxb, wyb, wzb = 1-wxu, 1-wyu, 1-wzu

  iuh, juh, kuh = ibh+1, jbh+1, kbh+1
  wxuh, wyuh, wzuh = x-ibh, y-jbh, z-kbh
  wxbh, wybh, wzbh = 1-wxuh, 1-wyuh, 1-wzuh

  ex[0] = interp_field_comp(E, 0, ibh,jb,kb, iuh,ju,ku,
                            wxbh,wyb,wzb, wxuh,wyu,wzu)

  ey[0] = interp_field_comp(E, 1, ib,jbh,kb, iu,juh,ku,
                            wxb,wybh,wzb, wxu,wyuh,wzu)

  ez[0] = interp_field_comp(E, 2, ib,jb,kbh, iu,ju,kuh,
                            wxb,wyb,wzbh, wxu,wyu,wzuh)

  bx[0] = interp_field_comp(B, 0, ib,jbh,kbh, iu,juh,kuh,
                            wxb,wybh,wzbh, wxu,wyuh,wzuh)

  by[0] = interp_field_comp(B, 1, ibh,jb,kbh, iuh,ju,kuh,
                            wxbh,wyb,wzbh, wxuh,wyu,wzuh)

  bz[0] = interp_field_comp(B, 2, ibh,jbh,kb, iuh,juh,ku,
                            wxbh,wybh,wzb, wxuh,wyuh,wzu)



# ==========================================================================

cdef void move_particles(BnzParticles prts, BnzGrid grid, double c):

  cdef:

    ints n

    GridParams gp = grid.params
    GridData gd = grid.data

    ParticleData *pd = &(prts.data)

    real ex,ey,ez, bx,by,bz
    real x,y,z, u,v,w, g, qm

    double dth, cinv


  cinv = 1./c
  ex,ey,ez, bx,by,bz = 0,0,0, 0,0,0

  for n in range(prts.Np):

    x,y,z   = pd.x[n], pd.y[n], pd.z[n]
    u,v,w,g = pd.u[n], pd.v[n], pd.w[n], pd.g[n]
    qm = prts.spc_prop[pd.spc[n]].qm

    interp_fields_to_loc(&ex, &ey, &ez, &bx, &by, &bz,
                         x,y,z, gd.E, gd.B, gp.ng)

    move_single_particle_vay(&x,&y,&z, &u,&v,&w,&g, qm,
                             ex,ey,ez, bx,by,bz, c,cinv)

    pd.x[n], pd.y[n], pd.z[n]          = x,y,z
    pd.u[n], pd.v[n], pd.w[n], pd.g[n] = u,v,w,g



# ==========================================================================

cdef inline real interp_field_comp(real4d A, int n,
                            ints ib, ints jb, ints kb,
                            ints iu, ints ju, ints ku,
                            real wxb, real wyb, real wzb,
                            real wxu, real wyu, real wzu) nogil:

    IF D2D and D3D:
      return ( wxb * wyb * wzb * A[n,kb,jb,ib]
             + wxb * wyb * wzu * A[n,ku,jb,ib]
             + wxb * wyu * wzb * A[n,kb,ju,ib]
             + wxb * wyu * wzu * A[n,ku,ju,ib]
             + wxu * wyb * wzb * A[n,kb,jb,iu]
             + wxu * wyb * wzu * A[n,ku,jb,iu]
             + wxu * wyu * wzb * A[n,kb,ju,iu]
             + wxu * wyu * wzu * A[n,ku,ju,iu] )
    ELIF D2D:
      return ( wxb * wyb * A[n,0,jb,ib]
             + wxb * wyu * A[n,0,ju,ib]
             + wxu * wyb * A[n,0,jb,iu]
             + wxu * wyu * A[n,0,ju,iu] )
    ELSE:
      return wxb * A[n,0,0,ib] + wxu * A[n,0,0,iu]
