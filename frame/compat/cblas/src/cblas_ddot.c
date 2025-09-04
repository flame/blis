#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_ddot.c
 *
 * The program is a C interface to ddot.
 * It calls the fortran wrapper before calling ddot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 * Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
double cblas_ddot( f77_int N, const double *X,
                      f77_int incX, const double *Y, f77_int incY)
{
   AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
   double dot;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif

   F77_ddot_sub( &F77_N, X, &F77_incX, Y, &F77_incY, &dot);
   AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
   return dot;
}
#endif
