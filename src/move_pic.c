#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void move(ParticleData *pd, real ****ec3, real ****bc3,
          ParticleProp *pp, int ng, real c) {

  real cinv = 1./c;

#if D2D
  int const DJW=2;
#else
  int const DJW=1;
#endif

#if D3D
  int const DKW=2;
#else
  int const DKW=1;
#endif

  // int ng=gc.ng;
  ints nprt = pp.Np;

  real bxv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real byv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real bzv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  real exv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real eyv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real ezv[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));


  for (ints n=0; n<nprt; n+=SIMD_WIDTH) {

    int rem = nprt - n;
    real *restrict xn = pd.x + n;
    real *restrict yn = pd.y + n;
    real *restrict zn = pd.z + n;
    real *restrict un = pd.u + n;
    real *restrict vn = pd.v + n;
    real *restrict wn = pd.w + n;
    real *restrict gn = pd.g + n;
    int  *restrict spcn = pd.spc + n;

    // Interpolate EM fields at predicted particle locations at time n+1/2.

#pragma omp simd aligned(xn,yn,zn:64) simdlen(SIMD_WIDTH)
    for (int nn=0; nn<IMIN(SIMD_WIDTH,rem); nn++) {

      real wgtx[2];
      real wgty[2];
      real wgtz[2];

      real wgtxh[2];
      real wgtyh[2];
      real wgtzh[2];

      int ib=0, jb=0, kb=0;
      int ibh=0, jbh=0, kbh=0;

      ib  = (int)FLOOR(x) + ng;
      ibh = (int)FLOOR(x-0.5) + ng;
      wgtx[1]  = x-ib;
      wgtxh[1] = x-ibh;
#if D2D
      jb  = (int)FLOOR(x) + ng;
      jbh = (int)FLOOR(y-0.5) + ng;
      wgty[1]  = y-jb;
      wgtyh[1] = y-jbh;
#endif
#if D3D
      kb  = (int)FLOOR(x) + ng;
      kbh = (int)FLOOR(z-0.5) + ng;
      wgtz[1]  = z-kb;
      wgtzh[1] = z-kbh;
#endif

      wgtx[0]  = 1.-wgtx[1];
      wgtxh[0] = 1.-wgtxh[1];

      wgty[0]  = 1.-wgty[1];
      wgtyh[0] = 1.-wgtyh[1];

      wgtz[0]  = 1.-wgtz[1];
      wgtzh[0] = 1.-wgtzh[1];

      exv[nn]=0.; eyv[nn]=0.; ezv[nn]=0.;
      bxv[nn]=0.; byv[nn]=0.; bzv[nn]=0.;

      for (int k0=0; k0<DKW; ++k0) {
        int k = kb + k0;
        int kh = kbh + k0;

        for (int j0=0; j0<DJW; ++j0) {
          int j = jb + j0;
          int jh = jbh + j0;

          for (int i0=0; i0<DIW; ++i0) {
            int i = ib + i0;
            int ih = ibh + i0;

            real wgt = wgtxh[i0] * wgty[j0] * wgtz[k0];
            exv[nn] += wgt * ec3[0][k][j][ih];

            wgt =  wgtx[i0] * wgtyh[j0] * wgtz[k0];
            eyv[nn] += wgt * ec3[1][k][jh][i];

            wgt =  wgtx[i0] * wgty[j0] * wgtzh[k0];
            ezv[nn] += wgt * ec3[2][kh][j][i];

            wgt =  wgtx[i0] * wgtyh[j0] * wgtzh[k0];
            bxv[nn] += wgt * bc3[0][kh][jh][i];

            wgt = wgtxh[i0] * wgty[j0] * wgtzh[k0];
            byv[nn] += wgt * bc3[1][kh][j][ih];

            wgt = wgtxh[i0] * wgtyh[j0] * wgtz[k0];
            bzv[nn] += wgt * bc3[2][k][jh][ih];


          }
        }
      }

    }

    // Update particle positions and velocities.

#pragma omp simd aligned(xn,yn,zn,un,vn,wn,gn,spcn:64) simdlen(SIMD_WIDTH)
    for (int nn=0; nn<IMIN(SIMD_WIDTH,rem); nn++) {

      // Vay pusher.

      real x = xn[nn];
      real y = yn[nn];
      real z = zn[nn];
      real u = un[nn];
      real v = vn[nn];
      real w = wn[nn];
      real g = gn[nn];

      real qm = pp.spc_props[spcn[nn]];

      real qmh = 0.5*qm;
      real qmh_c = qmh*cinv;
      real cinv2 = cinv*cinv;

      // dt=1

      real ex = qmh*ex;
      real ey = qmh*ey;
      real ez = qmh*ez;

      real taux = qmh_c * bx;
      real tauy = qmh_c * by;
      real tauz = qmh_c * bz;

      real tau2 = taux*taux + tauy*tauy + tauz*tauz;

      real ginv0 = 1. / SQRT(1. + cinv2 * (u*u + v*v + w*w));
      real vx0 = ginv0 * u;
      real vy0 = ginv0 * v;
      real vz0 = ginv0 * w;

      real u1 = u + 2.*ex + vx0 * tauz - vz0 * tauy;
      real v1 = v + 2.*ey + vz0 * taux - vx0 * tauz;
      real w1 = w + 2.*ez + vx0 * tauy - vy0 * taux;

      real g1 = SQRT(1. + cinv2 * (u1*u1 + v1*v1 + w1*w1));
      real sig = g1*g1 - tau2;
      real us = cinv * (u1 * taux + v1 * tauy + w1 * tauz);

      real gf = SQRT(0.5*(sig + SQRT(sig*sig + 4.*(tau2 + us*us))));
      real gfinv = 1./gf;

      real tx = gfinv * taux;
      real ty = gfinv * tauy;
      real tz = gfinv * tauz;

      real s = 1. / (1. + tau2 * gfinv*gfinv);
      real u1t = c*gfinv * us;

      real uf = s * (u1 + u1t * tx + v1 * tz - w1 * ty);
      real vf = s * (v1 + u1t * ty + w1 * tx - u1 * tz);
      real wf = s * (w1 + u1t * tz + u1 * ty - v1 * tx);

      real xf = x + gfinv * uf;
#if D2D
      real yf = y + gfinv * vf;
#else
      real yf=y;
#endif
#if D3D
      real zf = z + gfinv * wf;
#else
      real zf=z;
#endif

      xn[nn] = xf;
      yn[nn] = yf;
      zn[nn] = zf;
      un[nn] = uf;
      vn[nn] = vf;
      wn[nn] = wf;
      gn[nn] = gf;

    }

  }

}
