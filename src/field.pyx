# -*- coding: utf-8 -*-


import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange, threadid

from libc.math cimport sqrt,floor,log,exp,sin,cos,pow,fabs,fmin,fmax


# ===============================================================================

cdef void advance_b_field(real4d B, real4d E, ints lims[6], double dt, double c) nogil:

  cdef:
    ints i,j,k
    double cdt=c*dt

  IF D3D and D2D:

    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          B[0,k,j,i] = ( B[0,k,j,i] - cdt*(E[2,k,j+1,i] - E[2,k,j,i])
                                    + cdt*(E[1,k+1,j,i] - E[1,k,j,i]) )

          B[1,k,j,i] = ( B[1,k,j,i] + cdt*(E[2,k,j,i+1] - E[2,k,j,i])
                                    - cdt*(E[0,k+1,j,i] - E[0,k,j,i]) )

          B[2,k,j,i] = ( B[2,k,j,i] - cdt*(E[1,k,j,i+1] - E[1,k,j,i])
                                    + cdt*(E[0,k,j+1,i] - E[0,k,j,i]) )


  ELIF D2D:

    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        B[0,0,j,i] =   B[0,0,j,i] - cdt*(E[2,0,j+1,i] - E[2,0,j,i])

        B[1,0,j,i] =   B[1,0,j,i] + cdt*(E[2,0,j,i+1] - E[2,0,j,i])

        B[2,0,j,i] = ( B[2,0,j,i] - cdt*(E[1,0,j,i+1] - E[1,0,j,i])
                                  + cdt*(E[0,0,j+1,i] - E[0,0,j,i]) )

  ELSE:

    for i in range(lims[0],lims[1]+1):

      B[0,0,0,i] = B[0,0,0,i]
      B[1,0,0,i] = B[1,0,0,i] + cdt*(E[2,0,0,i+1] - E[2,0,0,i])
      B[2,0,0,i] = B[2,0,0,i] - cdt*(E[1,0,0,i+1] - E[1,0,0,i])



# ==============================================================================

cdef void advance_e_field(real4d E, real4d B, real4d J,
                          ints lims[6], double dt, double c) nogil:

  cdef:
    ints i,j,k
    double cdt=c*dt

  IF D2D and D3D:
    for k in range(lims[4],lims[5]+1):
      for j in range(lims[2],lims[3]+1):
        for i in range(lims[0],lims[1]+1):

          E[0,k,j,i] = ( E[0,k,j,i] + cdt * (B[2,k,j,i] - B[2,k,j-1,i])
                                    - cdt * (B[1,k,j,i] - B[1,k-1,j,i])
                       + J[0,k,j,i] )

          E[1,k,j,i] = ( E[1,k,j,i] - cdt * (B[2,k,j,i] - B[2,k,j,i-1])
                                    + cdt * (B[0,k,j,i] - B[0,k-1,j,i])
                       + J[1,k,j,i] )

          E[2,k,j,i] = ( E[2,k,j,i] + cdt * (B[1,k,j,i] - B[1,k,j,i-1])
                                    - cdt * (B[0,k,j,i] - B[0,k,j-1,i])
                       + J[2,k,j,i] )

  ELIF D2D:

    for j in range(lims[2],lims[3]+1):
      for i in range(lims[0],lims[1]+1):

        E[0,0,j,i] = ( E[0,0,j,i] + cdt * (B[2,0,j,i] - B[2,0,j-1,i])
                     + J[0,0,j,i] )

        E[1,0,j,i] = ( E[1,0,j,i] - cdt * (B[2,0,j,i] - B[2,0,j,i-1])
                     + J[1,0,j,i] )

        E[2,0,j,i] = ( E[2,0,j,i] + cdt * (B[1,0,j,i] - B[1,0,j,i-1])
                                  - cdt * (B[0,0,j,i] - B[0,0,j-1,i])
                     + J[2,0,j,i] )

  ELSE:

    for i in range(lims[0],lims[1]+1):

      E[0,0,0,i] = E[0,0,0,i] + J[0,0,0,i]

      E[1,0,0,i] = ( E[1,0,0,i] - cdt * (B[2,0,0,i] - B[2,0,0,i-1])
                   + J[1,0,0,i] )

      E[2,0,0,i] = ( E[2,0,0,i] + cdt * (B[1,0,0,i] - B[1,0,0,i-1])
                   + J[2,0,0,i] )
