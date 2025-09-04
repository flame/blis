#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_sdot.c
 *
 * The program is a C interface to sdot.
 * It calls the fortran wrapper before calling sdot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 * Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
float cblas_sdot( f77_int N, const float *X,
                      f77_int incX, const float *Y, f77_int incY)
{
   AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
   float dot;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif

   F77_sdot_sub( &F77_N, X, &F77_incX, Y, &F77_incY, &dot);
   AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
   return dot;
}
#endif
