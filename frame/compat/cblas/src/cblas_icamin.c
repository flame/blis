#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_icamin.c
 *
 * The program is a C interface to icamin.
 * It calls the fortran wrapper before calling icamin.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 */

#include "cblas.h"
#include "cblas_f77.h"
f77_int cblas_icamin( f77_int N, const void *X, f77_int incX)
{
   f77_int iamin;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else
   #define F77_N N
   #define F77_incX incX
#endif
   F77_icamin_sub( &F77_N, (scomplex*)X, &F77_incX, &iamin);
   return iamin ? iamin-1 : 0;
}
#endif
