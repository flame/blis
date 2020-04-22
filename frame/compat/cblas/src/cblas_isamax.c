#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_isamax.c
 *
 * The program is a C interface to isamax.
 * It calls the fortran wrapper before calling isamax.
 *
 * Written by Keita Teranishi.  2/11/1998
 * Copyright (C) 2020, Advanced Micro Devices, Inc.
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
f77_int cblas_isamax( f77_int N, const float *X, f77_int incX)
{
   f77_int iamax;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif

#ifdef BLIS_CONFIG_ZEN2
    dim_t    n0;
    float* x0;
    inc_t    incx0;
    gint_t   bli_index;

    /* If the vector is empty, return an index of zero. This early check
       is needed to emulate netlib BLAS. Without it, bli_?amaxv() will
       return 0, which ends up getting incremented to 1 (below) before
       being returned, which is not what we want. */
    if ( F77_N < 1 || F77_incX <= 0 ) return 0;

    /* Initialize BLIS. */
//  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( F77_N < 0 ) n0 = ( dim_t )0;
    else              n0 = ( dim_t )(F77_N);

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    if ( F77_incX < 0 )
    {
        /* The semantics of negative stride in BLAS are that the vector
        operand be traversed in reverse order. (Another way to think
        of this is that negative strides effectively reverse the order
        of the vector, but without any explicit data movements.) This
        is also how BLIS interprets negative strides. The differences
        is that with BLAS, the caller *always* passes in the 0th (i.e.,
        top-most or left-most) element of the vector, even when the
        stride is negative. By contrast, in BLIS, negative strides are
        used *relative* to the vector address as it is given. Thus, in
        BLIS, if this backwards traversal is desired, the caller *must*
        pass in the address to the (n-1)th (i.e., the bottom-most or
        right-most) element along with a negative stride. */

        x0    = ((float*)X) + (n0-1)*(-F77_incX);
        incx0 = ( inc_t )(F77_incX);

    }
    else
    {
        x0    = ((float*)X);
        incx0 = ( inc_t )(F77_incX);
    }

    /* Call BLIS kernel. */
    bli_samaxv_zen_int
    (
      n0,
      x0, incx0,
      &bli_index,
      NULL
    );

    /* Finalize BLIS. */
//    bli_finalize_auto();

    iamax = bli_index;

    return iamax;

#else
   F77_isamax_sub( &F77_N, X, &F77_incX, &iamax);
   return iamax ? iamax-1 : 0;
#endif
}
#endif
