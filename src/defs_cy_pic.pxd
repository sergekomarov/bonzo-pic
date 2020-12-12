# -*- coding: utf-8 -*-

# Definitions of types and external C functions.

cdef extern from "sys/types.h" nogil:
  ctypedef int time_t
  ctypedef int subseconds_t

cdef extern from "sys/time.h" nogil:
  cdef struct timeval:
    time_t tv_sec
    subseconds_t tv_usec
  int gettimeofday(timeval*, void*)


#---------------------------------------------------

# Type shortcuts.

# Use this type for indexing.
# ctypedef Py_ssize_t int

# Define "real" depending on precision.

IF SPREC:
  ctypedef float real
  # np_real = np.float32
  ctypedef float[::1] real1d
  ctypedef float[:,::1] real2d
  ctypedef float[:,:,::1] real3d
  ctypedef float[:,:,:,::1] real4d
  ctypedef float[:,:,:,:,::1] real5d
ELSE:
  ctypedef double real
  # np_real = np.float64
  ctypedef double[::1] real1d
  ctypedef double[:,::1] real2d
  ctypedef double[:,:,::1] real3d
  ctypedef double[:,:,:,::1] real4d
  ctypedef double[:,:,:,:,::1] real5d

ctypedef int[::1] int1d
ctypedef int[:,::1] int2d
ctypedef int[:,:,::1] int3d

#---------------------------------------------------

cdef extern from "defs.h" nogil:
  int NVAR
  int NPRT_PROP
  enum: XAX,YAX,ZAX
  enum: EX,EY,EZ, BX,BY,BZ, JX,JY,JZ
  real B_PI

#----------------------------------------------------

cdef extern from "defs.h" nogil:
  real SQR(real)
  real CUBE(real)
  real SQRT(real)
  real FABS(real)
  real FSIGN(real)
  real LOG(real)
  real POW(real)
  real EXP(real)
  real FLOOR(real)
  real FMIN(real)
  real FMAX(real)
  real IMIN(real)
  real IMAX(real)
  real SIN(real)
  real COS(real)
