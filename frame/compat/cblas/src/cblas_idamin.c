#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_idamin.c
 *
 * The program is a C interface to idamin.
 * It calls the fortran wrapper before calling idamin.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 */

#include "cblas.h"
#include "cblas_f77.h"
f77_int cblas_idamin( f77_int N, const double *X, f77_int incX)
{
   f77_int iamin;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else
   #define F77_N N
   #define F77_incX incX
#endif

   F77_idamin_sub( &F77_N, X, &F77_incX, &iamin);
   return iamin ? iamin-1 : 0;
}
#endif
