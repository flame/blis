#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_dcopy.c
 *
 * The program is a C interface to dcopy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc.
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dcopy( f77_int N, const double *X,
                      f77_int incX, double *Y, f77_int incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
#ifdef BLIS_CONFIG_ZEN2
    dim_t  n0;
    double* x0;
    double* y0;
    inc_t  incx0;
    inc_t  incy0;

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

        x0    = (double*)((X) + (n0-1)*(-F77_incX));
        incx0 = ( inc_t )(F77_incX);

    }
    else
    {
        x0    = (double*)(X);
        incx0 = ( inc_t )(F77_incX);
    }

    if ( F77_incY < 0 )
    {
        y0    = (Y) + (n0-1)*(-F77_incY);
        incy0 = ( inc_t )(F77_incY);

    }
    else
    {
        y0    = (Y);
        incy0 = ( inc_t )(F77_incY);
    }


    /* Call BLIS kernel */
    bli_dcopyv_zen_int
    (
	    BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        NULL
    );

    /* Finalize BLIS. */
//    bli_finalize_auto();
#else
   F77_dcopy( &F77_N, X, &F77_incX, Y, &F77_incY);
#endif
   
}
#endif
