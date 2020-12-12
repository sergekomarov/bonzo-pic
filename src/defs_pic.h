#include <stdint.h>
#include <math.h>

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// fixed-length integer type
// typedef intptr_t ints;
// floating-point type depending on precision
#if SPREC
typedef float real;
#else
typedef double real;
#endif

#if !defined(CACHELINE_BYTES)
#define CACHELINE_BYTES 64
#endif

#if defined(__AVX512F__)
#define SIMD_WIDTH 8
#elif defined(__AVX__)
#define SIMD_WIDTH 4
#elif defined(__AVX2__)
#define SIMD_WIDTH 4
#elif defined(__SSE2__)
#define SIMD_WIDTH 2
#else
#define SIMD_WIDTH 4
#endif


//===========================================================

#define NVAR 9
#define NPRT_PROP 10
// indices of electromagnetic fields and particle currents
enum {EX=0,EY,EZ, BX,BY,BZ, JX,JY,JZ};

// names of coordinate axes
enum {XAX,YAX,ZAX};


// ====================================================================

#define B_PI 3.14159265
#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
#define IMIN(a,b) ( ((a) < (b)) ? (a) : (b) )
#define IMAX(a,b) ( ((a) > (b)) ? (a) : (b) )

#if SPREC

#define FMAX(a) ( fmaxf(a) )
#define FMIN(a) ( fminf(a) )
#define FLOOR(a) ( floorf(a) )
#define FABS(a)  ( fabsf(a) )
#define POW(a)  ( powf(a) )
#define LOG(a)  ( logf(a) )
#define EXP(a)  ( expf(a) )
#define SQRT(a) ( sqrtf(a) )
#define SIGN(a) ( ( 0.0f < (a) ) - ((a) < 0.0f) )
#define SIN(a) ( sinf(a) )
#define COS(a) ( cosf(a) )

#else

#define FMAX(a) ( fmax(a) )
#define FMIN(a) ( fmin(a) )
#define FLOOR(a) ( floor(a) )
#define FABS(a)  ( fabs(a) )
#define POW(a)  ( pow(a) )
#define LOG(a)  ( log(a) )
#define EXP(a)  ( exp(a) )
#define SQRT(a) ( sqrt(a) )
#define SIGN(a) ( ( 0.0 < (a) ) - ((a) < 0.0) )
#define SIN(a) ( sin(a) )
#define COS(a) ( cos(a) )

#endif


#endif
