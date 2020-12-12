#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void deposit(real ****cur, ParticleData *pd,
             ParticleProp *pp, GridCoord *gc,
             real c) {

  real cinv2 = 1./(c*c);

  int const DIW=2;

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

  int ngridx=gc.Ntot[0], ngridy=gc.Ntot[1], ngridz=gc.Ntot[2], ng=gc.ng;
  ints nprt = pp.Np;

  real curxv[2][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real curyv[2][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  real curzv[2][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  int wgtyzv[2][4][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int wgtxzv[2][4][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int wgtxyv[2][4][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  int ibv[2][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int jbv[2][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  int kbv[2][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  real *curv = (real*)calloc(12*ngridx*ngridy*ngridz, sizeof(real));


  for (ints n=0; n<nprt; n+=SIMD_WIDTH) {

    int rem = nprt - n;
    real *restrict xn = pd.x + n;
    real *restrict yn = pd.y + n;
    real *restrict zn = pd.z + n;
    real *restrict un = pd.u + n;
    real *restrict vn = pd.v + n;
    real *restrict wn = pd.w + n;
    real *restrict gn = pd.g + n;
    real *restrict mn = pd.m + n;
    int  *restrict spcn = pd.spc + n;

    // Current deposition based on Umeda 2003.

#pragma omp simd aligned(xn,yn,zn,un,vn,wn,gn,mn,spcn:64) simdlen(SIMD_WIDTH)
    for (int nn=0; nn<IMIN(SIMD_WIDTH,rem); nn++) {

      real wgtx[2][2];
      real wgty[2][2];
      real wgtz[2][2];

      real x2 = xn[nn];
      real y2 = yn[nn];
      real z2 = zn[nn];

      real u = un[nn];
      real v = vn[nn];
      real w = wn[nn];
      real g = gn[nn];

      real mass = mn[nn];
      real spc = spcn[nn];

      real ginv = 1. / g;
      real qp = mass * pp.spc_props[spc].qm;

      // x coordinate

      real x1 = x2 - ginv * u;

      real ib1 = (int)FLOOR(x1);
      real ib2 = (int)FLOOR(x2);

      // coordinates of the relay point (Umeda 2003)
      real xr = FMIN(FMIN(ib1,ib2) + 1., FMAX(FMAX(ib1,ib2), 0.5*(x1+x2)));

      // charge fluxes
      // dt=1 for pure PIC, q/dt=q
      curxv[0][nn] = qp * (xr-x1);
      curxv[1][nn] = qp * (x2-xr);

      // weights for charge flux interpolation
      wgtx[0][1] = 0.5 * (x1 + xr) - ib1;
      wgtx[1][1] = 0.5 * (xr + x2) - ib2;

      ibv[0][nn] = ib1 + ng;
      ibv[1][nn] = ib2 + ng;

#if D2D:
      // y coordinate

      real y1 = y2 - ginv * v;

      real jb1 = (int)FLOOR(y1);
      real jb2 = (int)FLOOR(y2);

      real yr = FMIN(FMIN(jb1,jb2) + 1., FMAX(FMAX(jb1,jb2), 0.5*(y1+y2)));

      curyv[0][nn] = qp * (yr-y1);
      curyv[1][nn] = qp * (y2-yr);

      wgty[0][1] = 0.5 * (y1 + yr) - jb1;
      wgty[1][1] = 0.5 * (yr + y2) - jb2;

      jbv[0][nn] = jb1 + ng;
      jbv[1][nn] = jb2 + ng;
#endif

#if D3D
      // z coordinate

      real z1 = z2 - ginv * w;

      real kb1 = (int)FLOOR(z1);
      real kb2 = (int)FLOOR(z2);

      real zr = FMIN(FMIN(kb1,kb2) + 1., FMAX(FMAX(kb1,kb2), 0.5*(z1+z2)));

      curzv[0][nn] = qp * (zr-z1);
      curzv[1][nn] = qp * (z2-zr);

      wgtz[0][1] = 0.5 * (z1 + zr) - kb1;
      wgtz[1][1] = 0.5 * (zr + z2) - kb2;

      kbv[0][nn] = kb1 + ng;
      kbv[1][nn] = kb2 + ng;
#endif

      for (int d=0; d<2; ++d) {

        wgtx[d][0] = 1.-wgtx[d][1];
        wgty[d][0] = 1.-wgty[d][1];
        wgtz[d][0] = 1.-wgtz[d][1];

        for (int k=0; k<DKW; ++k) {
          for (int j=0; j<DJW; ++j) {
            int m = j + k*DJW;
            wgtyzv[d][m][nn] = wgty[d][j] * wgtz[d][k];
          }
        }

        for (int k=0; k<DKW; ++k) {
          for (int i=0; i<DIW; ++i) {
            int m = i + k*DIW;
            wgtxzv[d][m][nn] = wgtx[d][i] * wgtz[d][k];
          }
        }

        for (int j=0; j<DJW; ++j) {
          for (int i=0; i<DIW; ++i) {
            int m = i + j*DIW;
            wgtxyv[d][m][nn] = wgtx[d][i] * wgty[d][j];
          }
        }

      }

    }

    // add deposits to temporary vectorized 1D array curv

    for (int nn=0; nn<IMIN(SIMD_WIDTH,rem); nn++) {

      real curx[2];
      real cury[2];
      real curz[2];
      int ind[2];

      for (int d=0; d<2; ++d) {

        curx[d] = curxv[d][nn];
        cury[d] = curyv[d][nn];
        curz[d] = curzv[d][nn];

        ind[d] = ibv[d][nn] + jbv[d][nn]*ngridx + kbv[d][nn]*ngridx*ngridy;

#if D3D

        real *curvn = curv + 12*ind[d];
#pragma omp simd aligned(curvn) simdlen(IMIN(SIMD_WIDTH,4))
        for (int i=0; i<4; i++) {
          curvn[i]   += wyzv[d][i][nn] * curx[d];
          curvn[i+4] += wxzv[d][i][nn] * cury[d];
          curvn[i+8] += wxyv[d][i][nn] * curz[d];
        }

#elif D2D

        real *curvn = curv + 8*ind[d];
#pragma omp simd aligned(curvn) simdlen(IMIN(SIMD_WIDTH,2))
        for (int i=0; i<2; i++) {
          curvn[i] += wyzv[d][i][nn] * curx[d];
          curvn[i+2] += wxzv[d][i][nn] * cury[d];
        }

#pragma omp simd aligned(curvn) simdlen(IMIN(SIMD_WIDTH,4))
        for (int i=0; i<4; i++)
          curvn[i+4] += wxyv[d][i][nn] * curz[d];

#else

        real *curvn = curv + 5*ind[d];
        curvn[0] += curx[0];

#pragma omp simd aligned(cur1vn) simdlen(IMIN(SIMD_WIDTH,2))
        for (int i=0; i<2; i++) {
          real wx1 = wxzv[d][i][nn];
          curvn[i+1] += wx1 * cury[d];
          curvn[i+3] += wx1 * curz[d];

        }
#endif

      }
    }

  }

  // add deposits from temporary arrayz to the main 3D current array

  for (int k=0; k<ngridz-1; ++k) {
    for (int j=0; j<ngridy-1; ++j) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=0; i<ngridx-1; ++i) {

#if D3D
        int ind = 12 * (i + j*ngridx + k*ngridx*ngridy);

        cur[0][k][j][i] += curv[ind];
        cur[0][k][j+1][i] += curv[ind+1];
        cur[0][k+1][j][i] += curv[ind+2];
        cur[0][k+1][j+1][i] += curv[ind+3];

        cur[1][k][j][i] += curv[ind+4];
        cur[1][k][j][i+1] += curv[ind+5];
        cur[1][k+1][j][i] += curv[ind+6];
        cur[1][k+1][j][i+1] += curv[ind+7];

        cur[2][k][j][i] += curv[ind+8];
        cur[2][k][j][i+1] += curv[ind+9];
        cur[2][k][j+1][i] += curv[ind+10];
        cur[2][k][j+1][i+1] += curv[ind+11];

#elif D2D
        int ind = 8 * (i + j*ngridx + k*ngridx*ngridy);

        cur[0][k][j][i] += curv[ind];
        cur[0][k][j+1][i] += curv[ind+1];

        cur[1][k][j][i] += curv[ind+2];
        cur[1][k][j][i+1] += curv[ind+3];

        cur[2][k][j][i] += curv[ind+4];
        cur[2][k][j][i+1] += curv[ind+5];
        cur[2][k][j+1][i] += curv[ind+6];
        cur[2][k][j+1][i+1] += curv[ind+7];
#else
        int ind = 5 * (i + j*ngridx + k*ngridx*ngridy);

        cur[0][k][j][i] += curv[ind];

        cur[1][k][j][i] += curv[ind+1];
        cur[1][k][j][i+1] += curv[ind+2];

        cur[2][k][j][i] += curv[ind+3];
        cur[2][k][j][i+1] += curv[ind+4];

      }
    }
  }

}
