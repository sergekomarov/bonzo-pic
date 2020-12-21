# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

cdef void advance_bfld(real4d bfld, real4d efld, int lims[6], real dt, real c) nogil:

  cdef:
    int i,j,k
    real cdt=c*dt

  IF D3D and D2D:

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          bfld[0,k,j,i] = ( bfld[0,k,j,i] - cdt*(efld[2,k,j+1,i] - efld[2,k,j,i])
                                          + cdt*(efld[1,k+1,j,i] - efld[1,k,j,i]) )

          bfld[1,k,j,i] = ( bfld[1,k,j,i] + cdt*(efld[2,k,j,i+1] - efld[2,k,j,i])
                                          - cdt*(efld[0,k+1,j,i] - efld[0,k,j,i]) )

          bfld[2,k,j,i] = ( bfld[2,k,j,i] - cdt*(efld[1,k,j,i+1] - efld[1,k,j,i])
                                          + cdt*(efld[0,k,j+1,i] - efld[0,k,j,i]) )


  ELIF D2D:

    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        bfld[0,0,j,i] =   bfld[0,0,j,i] - cdt*(efld[2,0,j+1,i] - efld[2,0,j,i])

        bfld[1,0,j,i] =   bfld[1,0,j,i] + cdt*(efld[2,0,j,i+1] - efld[2,0,j,i])

        bfld[2,0,j,i] = ( bfld[2,0,j,i] - cdt*(efld[1,0,j,i+1] - efld[1,0,j,i])
                                        + cdt*(efld[0,0,j+1,i] - efld[0,0,j,i]) )

  ELSE:

    for i in range(lims[0],lims[1]+1):

      bfld[0,0,0,i] = bfld[0,0,0,i]
      bfld[1,0,0,i] = bfld[1,0,0,i] + cdt*(efld[2,0,0,i+1] - efld[2,0,0,i])
      bfld[2,0,0,i] = bfld[2,0,0,i] - cdt*(efld[1,0,0,i+1] - efld[1,0,0,i])


# ---------------------------------------------------------------------------

cdef void advance_efld(real4d efld, real4d bfld, real4d curr,
                       int lims[6], real dt, real c) nogil:

  cdef:
    int i,j,k
    real cdt=c*dt

  IF D2D and D3D:
    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          efld[0,k,j,i] = ( efld[0,k,j,i] + cdt * (bfld[2,k,j,i] - bfld[2,k,j-1,i])
                                          - cdt * (bfld[1,k,j,i] - bfld[1,k-1,j,i])
                       + curr[0,k,j,i] )

          efld[1,k,j,i] = ( efld[1,k,j,i] - cdt * (bfld[2,k,j,i] - bfld[2,k,j,i-1])
                                          + cdt * (bfld[0,k,j,i] - bfld[0,k-1,j,i])
                       + curr[1,k,j,i] )

          efld[2,k,j,i] = ( efld[2,k,j,i] + cdt * (bfld[1,k,j,i] - bfld[1,k,j,i-1])
                                          - cdt * (bfld[0,k,j,i] - bfld[0,k,j-1,i])
                       + curr[2,k,j,i] )

  ELIF D2D:

    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        efld[0,0,j,i] = ( efld[0,0,j,i] + cdt * (bfld[2,0,j,i] - bfld[2,0,j-1,i])
                     + curr[0,0,j,i] )

        efld[1,0,j,i] = ( efld[1,0,j,i] - cdt * (bfld[2,0,j,i] - bfld[2,0,j,i-1])
                     + curr[1,0,j,i] )

        efld[2,0,j,i] = ( efld[2,0,j,i] + cdt * (bfld[1,0,j,i] - bfld[1,0,j,i-1])
                                        - cdt * (bfld[0,0,j,i] - bfld[0,0,j-1,i])
                     + curr[2,0,j,i] )

  ELSE:

    for i in range(lims[0],lims[1]+1):

      efld[0,0,0,i] = efld[0,0,0,i] + curr[0,0,0,i]

      efld[1,0,0,i] = ( efld[1,0,0,i] - cdt * (bfld[2,0,0,i] - bfld[2,0,0,i-1])
                   + curr[1,0,0,i] )

      efld[2,0,0,i] = ( efld[2,0,0,i] + cdt * (bfld[1,0,0,i] - bfld[1,0,0,i-1])
                   + curr[2,0,0,i] )
