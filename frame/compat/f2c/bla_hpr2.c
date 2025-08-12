/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

// Make thread settings local to each thread calling BLIS routines.
// (The definition resides in bli_rntm.c.)
extern BLIS_THREAD_LOCAL rntm_t tl_rntm;

/* chpr2.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ 
int PASTEF77S(c,hpr2)(const bla_character *uplo, const bla_integer *n, const bla_scomplex *alpha, const bla_scomplex *x, const bla_integer *incx, const bla_scomplex *y, const bla_integer *incy, bla_scomplex *ap)
{
    /* System generated locals */
    bla_integer i__1, i__2, i__3, i__4, i__5, i__6;
    bla_real r__1;
    bla_scomplex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    //void bla_r_cnjg(bla_scomplex *, bla_scomplex *);

    /* Local variables */
    bla_integer info;
    bla_scomplex temp1, temp2;
    bla_integer i__, j, k;
    //extern bla_logical PASTE_LSAME(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kk, ix, iy, jx = 0, jy = 0, kx = 0, ky = 0;
    //extern /* Subroutine */ int PASTE_XERBLA(bla_character *, bla_integer *, ftnlen);

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHPR2  performs the hermitian rank 2 operation */

/*     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A, */

/*  where alpha is a scalar, x and y are n element vectors and A is an */
/*  n by n hermitian matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the matrix A is supplied in the packed */
/*           array AP as follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  supplied in AP. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  supplied in AP. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX         . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the n */
/*           element vector y. */
/*           Unchanged on exit. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX          array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 ) */
/*           and a( 2, 2 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the upper triangular part of the */
/*           updated matrix. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 ) */
/*           and a( 3, 1 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the lower triangular part of the */
/*           updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --y;
    --x;

    /* Function Body */
    AOCL_DTL_INITIALIZE(AOCL_DTL_TRACE_LEVEL);
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_HPR2_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(c), *uplo,
		       	     *n, (void*)alpha, *incx, *incy);

    // Initialize info_value to 0
    gint_t info_value = 0;
    bli_rntm_set_info_value_only( info_value, &tl_rntm );

    info = 0;
    if (! PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    }
    if (info != 0) {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	PASTE_XERBLA("CHPR2 ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (bli_creal(*alpha) == 0.f && bli_cimag(*alpha) == 0.f)) {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	return 0;
    }

/*     Set up the start points in X and Y if the increments are not both */
/*     unity. */

    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }

/*     Start the operations. In this version the elements of the array AP */
/*     are accessed sequentially with one pass through AP. */

    kk = 1;
    if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f || (bli_creal(y[i__3]) != 0.f 
			|| bli_cimag(y[i__3]) != 0.f)) {
		    bla_r_cnjg(&q__2, &y[j]);
		    bli_csets( (bli_creal(*alpha) * bli_creal(q__2) - bli_cimag(*alpha) * bli_cimag(q__2)), (bli_creal(*alpha) * bli_cimag(q__2) + bli_cimag(*alpha) * bli_creal(q__2)), q__1 );
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp1 );
		    i__2 = j;
		    bli_csets( (bli_creal(*alpha) * bli_creal(x[i__2]) - bli_cimag(*alpha) * bli_cimag(x[i__2])), (bli_creal(*alpha) * bli_cimag(x[i__2]) + bli_cimag(*alpha) * bli_creal(x[i__2])), q__2 );
		    bla_r_cnjg(&q__1, &q__2);
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp2 );
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			bli_csets( (bli_creal(x[i__5]) * bli_creal(temp1) - bli_cimag(x[i__5]) * bli_cimag(temp1)), (bli_creal(x[i__5]) * bli_cimag(temp1) + bli_cimag(x[i__5]) * bli_creal(temp1)), q__3 );
			bli_csets( (bli_creal(ap[i__4]) + bli_creal(q__3)), (bli_cimag(ap[i__4]) + bli_cimag(q__3)), q__2 );
			i__6 = i__;
			bli_csets( (bli_creal(y[i__6]) * bli_creal(temp2) - bli_cimag(y[i__6]) * bli_cimag(temp2)), (bli_creal(y[i__6]) * bli_cimag(temp2) + bli_cimag(y[i__6]) * bli_creal(temp2)), q__4 );
			bli_csets( (bli_creal(q__2) + bli_creal(q__4)), (bli_cimag(q__2) + bli_cimag(q__4)), q__1 );
			bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), ap[i__3] );
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    bli_csets( (bli_creal(x[i__4]) * bli_creal(temp1) - bli_cimag(x[i__4]) * bli_cimag(temp1)), (bli_creal(x[i__4]) * bli_cimag(temp1) + bli_cimag(x[i__4]) * bli_creal(temp1)), q__2 );
		    i__5 = j;
		    bli_csets( (bli_creal(y[i__5]) * bli_creal(temp2) - bli_cimag(y[i__5]) * bli_cimag(temp2)), (bli_creal(y[i__5]) * bli_cimag(temp2) + bli_cimag(y[i__5]) * bli_creal(temp2)), q__3 );
		    bli_csets( (bli_creal(q__2) + bli_creal(q__3)), (bli_cimag(q__2) + bli_cimag(q__3)), q__1 );
		    r__1 = bli_creal(ap[i__3]) + bli_creal(q__1);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = bli_creal(ap[i__3]);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f || (bli_creal(y[i__3]) != 0.f 
			|| bli_cimag(y[i__3]) != 0.f)) {
		    bla_r_cnjg(&q__2, &y[jy]);
		    bli_csets( (bli_creal(*alpha) * bli_creal(q__2) - bli_cimag(*alpha) * bli_cimag(q__2)), (bli_creal(*alpha) * bli_cimag(q__2) + bli_cimag(*alpha) * bli_creal(q__2)), q__1 );
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp1 );
		    i__2 = jx;
		    bli_csets( (bli_creal(*alpha) * bli_creal(x[i__2]) - bli_cimag(*alpha) * bli_cimag(x[i__2])), (bli_creal(*alpha) * bli_cimag(x[i__2]) + bli_cimag(*alpha) * bli_creal(x[i__2])), q__2 );
		    bla_r_cnjg(&q__1, &q__2);
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp2 );
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			bli_csets( (bli_creal(x[i__5]) * bli_creal(temp1) - bli_cimag(x[i__5]) * bli_cimag(temp1)), (bli_creal(x[i__5]) * bli_cimag(temp1) + bli_cimag(x[i__5]) * bli_creal(temp1)), q__3 );
			bli_csets( (bli_creal(ap[i__4]) + bli_creal(q__3)), (bli_cimag(ap[i__4]) + bli_cimag(q__3)), q__2 );
			i__6 = iy;
			bli_csets( (bli_creal(y[i__6]) * bli_creal(temp2) - bli_cimag(y[i__6]) * bli_cimag(temp2)), (bli_creal(y[i__6]) * bli_cimag(temp2) + bli_cimag(y[i__6]) * bli_creal(temp2)), q__4 );
			bli_csets( (bli_creal(q__2) + bli_creal(q__4)), (bli_cimag(q__2) + bli_cimag(q__4)), q__1 );
			bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), ap[i__3] );
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    bli_csets( (bli_creal(x[i__4]) * bli_creal(temp1) - bli_cimag(x[i__4]) * bli_cimag(temp1)), (bli_creal(x[i__4]) * bli_cimag(temp1) + bli_cimag(x[i__4]) * bli_creal(temp1)), q__2 );
		    i__5 = jy;
		    bli_csets( (bli_creal(y[i__5]) * bli_creal(temp2) - bli_cimag(y[i__5]) * bli_cimag(temp2)), (bli_creal(y[i__5]) * bli_cimag(temp2) + bli_cimag(y[i__5]) * bli_creal(temp2)), q__3 );
		    bli_csets( (bli_creal(q__2) + bli_creal(q__3)), (bli_cimag(q__2) + bli_cimag(q__3)), q__1 );
		    r__1 = bli_creal(ap[i__3]) + bli_creal(q__1);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = bli_creal(ap[i__3]);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		}
		jx += *incx;
		jy += *incy;
		kk += j;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when lower triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f || (bli_creal(y[i__3]) != 0.f 
			|| bli_cimag(y[i__3]) != 0.f)) {
		    bla_r_cnjg(&q__2, &y[j]);
		    bli_csets( (bli_creal(*alpha) * bli_creal(q__2) - bli_cimag(*alpha) * bli_cimag(q__2)), (bli_creal(*alpha) * bli_cimag(q__2) + bli_cimag(*alpha) * bli_creal(q__2)), q__1 );
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp1 );
		    i__2 = j;
		    bli_csets( (bli_creal(*alpha) * bli_creal(x[i__2]) - bli_cimag(*alpha) * bli_cimag(x[i__2])), (bli_creal(*alpha) * bli_cimag(x[i__2]) + bli_cimag(*alpha) * bli_creal(x[i__2])), q__2 );
		    bla_r_cnjg(&q__1, &q__2);
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp2 );
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    bli_csets( (bli_creal(x[i__4]) * bli_creal(temp1) - bli_cimag(x[i__4]) * bli_cimag(temp1)), (bli_creal(x[i__4]) * bli_cimag(temp1) + bli_cimag(x[i__4]) * bli_creal(temp1)), q__2 );
		    i__5 = j;
		    bli_csets( (bli_creal(y[i__5]) * bli_creal(temp2) - bli_cimag(y[i__5]) * bli_cimag(temp2)), (bli_creal(y[i__5]) * bli_cimag(temp2) + bli_cimag(y[i__5]) * bli_creal(temp2)), q__3 );
		    bli_csets( (bli_creal(q__2) + bli_creal(q__3)), (bli_cimag(q__2) + bli_cimag(q__3)), q__1 );
		    r__1 = bli_creal(ap[i__3]) + bli_creal(q__1);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			bli_csets( (bli_creal(x[i__5]) * bli_creal(temp1) - bli_cimag(x[i__5]) * bli_cimag(temp1)), (bli_creal(x[i__5]) * bli_cimag(temp1) + bli_cimag(x[i__5]) * bli_creal(temp1)), q__3 );
			bli_csets( (bli_creal(ap[i__4]) + bli_creal(q__3)), (bli_cimag(ap[i__4]) + bli_cimag(q__3)), q__2 );
			i__6 = i__;
			bli_csets( (bli_creal(y[i__6]) * bli_creal(temp2) - bli_cimag(y[i__6]) * bli_cimag(temp2)), (bli_creal(y[i__6]) * bli_cimag(temp2) + bli_cimag(y[i__6]) * bli_creal(temp2)), q__4 );
			bli_csets( (bli_creal(q__2) + bli_creal(q__4)), (bli_cimag(q__2) + bli_cimag(q__4)), q__1 );
			bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), ap[i__3] );
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = bli_creal(ap[i__3]);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (bli_creal(x[i__2]) != 0.f || bli_cimag(x[i__2]) != 0.f || (bli_creal(y[i__3]) != 0.f 
			|| bli_cimag(y[i__3]) != 0.f)) {
		    bla_r_cnjg(&q__2, &y[jy]);
		    bli_csets( (bli_creal(*alpha) * bli_creal(q__2) - bli_cimag(*alpha) * bli_cimag(q__2)), (bli_creal(*alpha) * bli_cimag(q__2) + bli_cimag(*alpha) * bli_creal(q__2)), q__1 );
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp1 );
		    i__2 = jx;
		    bli_csets( (bli_creal(*alpha) * bli_creal(x[i__2]) - bli_cimag(*alpha) * bli_cimag(x[i__2])), (bli_creal(*alpha) * bli_cimag(x[i__2]) + bli_cimag(*alpha) * bli_creal(x[i__2])), q__2 );
		    bla_r_cnjg(&q__1, &q__2);
		    bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), temp2 );
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    bli_csets( (bli_creal(x[i__4]) * bli_creal(temp1) - bli_cimag(x[i__4]) * bli_cimag(temp1)), (bli_creal(x[i__4]) * bli_cimag(temp1) + bli_cimag(x[i__4]) * bli_creal(temp1)), q__2 );
		    i__5 = jy;
		    bli_csets( (bli_creal(y[i__5]) * bli_creal(temp2) - bli_cimag(y[i__5]) * bli_cimag(temp2)), (bli_creal(y[i__5]) * bli_cimag(temp2) + bli_cimag(y[i__5]) * bli_creal(temp2)), q__3 );
		    bli_csets( (bli_creal(q__2) + bli_creal(q__3)), (bli_cimag(q__2) + bli_cimag(q__3)), q__1 );
		    r__1 = bli_creal(ap[i__3]) + bli_creal(q__1);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			bli_csets( (bli_creal(x[i__5]) * bli_creal(temp1) - bli_cimag(x[i__5]) * bli_cimag(temp1)), (bli_creal(x[i__5]) * bli_cimag(temp1) + bli_cimag(x[i__5]) * bli_creal(temp1)), q__3 );
			bli_csets( (bli_creal(ap[i__4]) + bli_creal(q__3)), (bli_cimag(ap[i__4]) + bli_cimag(q__3)), q__2 );
			i__6 = iy;
			bli_csets( (bli_creal(y[i__6]) * bli_creal(temp2) - bli_cimag(y[i__6]) * bli_cimag(temp2)), (bli_creal(y[i__6]) * bli_cimag(temp2) + bli_cimag(y[i__6]) * bli_creal(temp2)), q__4 );
			bli_csets( (bli_creal(q__2) + bli_creal(q__4)), (bli_cimag(q__2) + bli_cimag(q__4)), q__1 );
			bli_csets( (bli_creal(q__1)), (bli_cimag(q__1)), ap[i__3] );
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = bli_creal(ap[i__3]);
		    bli_csets( (r__1), (0.f), ap[i__2] );
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return 0;

/*     End of CHPR2 . */

} /* chpr2_ */

/* zhpr2.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ 
int PASTEF77S(z,hpr2)(const bla_character *uplo, const bla_integer *n, const bla_dcomplex *alpha, const bla_dcomplex *x, const bla_integer *incx, const bla_dcomplex *y, const bla_integer *incy, bla_dcomplex *ap)
{
    /* System generated locals */
    bla_integer i__1, i__2, i__3, i__4, i__5, i__6;
    bla_double d__1;
    bla_dcomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    //void bla_d_cnjg(bla_dcomplex *, bla_dcomplex *);

    /* Local variables */
    bla_integer info;
    bla_dcomplex temp1, temp2;
    bla_integer i__, j, k;
    //extern bla_logical PASTE_LSAME(bla_character *, bla_character *, ftnlen, ftnlen);
    bla_integer kk, ix, iy, jx = 0, jy = 0, kx = 0, ky = 0;
    //extern /* Subroutine */ int PASTE_XERBLA(bla_character *, bla_integer *, ftnlen);

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZHPR2  performs the hermitian rank 2 operation */

/*     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A, */

/*  where alpha is a scalar, x and y are n element vectors and A is an */
/*  n by n hermitian matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the matrix A is supplied in the packed */
/*           array AP as follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  supplied in AP. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  supplied in AP. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX*16      . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the n */
/*           element vector y. */
/*           Unchanged on exit. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX*16       array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 ) */
/*           and a( 2, 2 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the upper triangular part of the */
/*           updated matrix. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 ) */
/*           and a( 3, 1 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the lower triangular part of the */
/*           updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --y;
    --x;

    /* Function Body */
    AOCL_DTL_INITIALIZE(AOCL_DTL_TRACE_LEVEL);
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_HPR2_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(z), *uplo,
		       	     *n, (void*)alpha, *incx, *incy);

    // Initialize info_value to 0
    gint_t info_value = 0;
    bli_rntm_set_info_value_only( info_value, &tl_rntm );

    info = 0;
    if (! PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTE_LSAME(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    }
    if (info != 0) {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	PASTE_XERBLA("ZHPR2 ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (bli_zreal(*alpha) == 0. && bli_zimag(*alpha) == 0.)) {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	return 0;
    }

/*     Set up the start points in X and Y if the increments are not both */
/*     unity. */

    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }

/*     Start the operations. In this version the elements of the array AP */
/*     are accessed sequentially with one pass through AP. */

    kk = 1;
    if (PASTE_LSAME(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0. || (bli_zreal(y[i__3]) != 0. || 
			bli_zimag(y[i__3]) != 0.)) {
		    bla_d_cnjg(&z__2, &y[j]);
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(z__2) - bli_zimag(*alpha) * bli_zimag(z__2)), (bli_zreal(*alpha) * bli_zimag(z__2) + bli_zimag(*alpha) * bli_zreal(z__2)), z__1 );
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp1 );
		    i__2 = j;
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(x[i__2]) - bli_zimag(*alpha) * bli_zimag(x[i__2])), (bli_zreal(*alpha) * bli_zimag(x[i__2]) + bli_zimag(*alpha) * bli_zreal(x[i__2])), z__2 );
		    bla_d_cnjg(&z__1, &z__2);
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp2 );
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			bli_zsets( (bli_zreal(x[i__5]) * bli_zreal(temp1) - bli_zimag(x[i__5]) * bli_zimag(temp1)), (bli_zreal(x[i__5]) * bli_zimag(temp1) + bli_zimag(x[i__5]) * bli_zreal(temp1)), z__3 );
			bli_zsets( (bli_zreal(ap[i__4]) + bli_zreal(z__3)), (bli_zimag(ap[i__4]) + bli_zimag(z__3)), z__2 );
			i__6 = i__;
			bli_zsets( (bli_zreal(y[i__6]) * bli_zreal(temp2) - bli_zimag(y[i__6]) * bli_zimag(temp2)), (bli_zreal(y[i__6]) * bli_zimag(temp2) + bli_zimag(y[i__6]) * bli_zreal(temp2)), z__4 );
			bli_zsets( (bli_zreal(z__2) + bli_zreal(z__4)), (bli_zimag(z__2) + bli_zimag(z__4)), z__1 );
			bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), ap[i__3] );
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    bli_zsets( (bli_zreal(x[i__4]) * bli_zreal(temp1) - bli_zimag(x[i__4]) * bli_zimag(temp1)), (bli_zreal(x[i__4]) * bli_zimag(temp1) + bli_zimag(x[i__4]) * bli_zreal(temp1)), z__2 );
		    i__5 = j;
		    bli_zsets( (bli_zreal(y[i__5]) * bli_zreal(temp2) - bli_zimag(y[i__5]) * bli_zimag(temp2)), (bli_zreal(y[i__5]) * bli_zimag(temp2) + bli_zimag(y[i__5]) * bli_zreal(temp2)), z__3 );
		    bli_zsets( (bli_zreal(z__2) + bli_zreal(z__3)), (bli_zimag(z__2) + bli_zimag(z__3)), z__1 );
		    d__1 = bli_zreal(ap[i__3]) + bli_zreal(z__1);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    d__1 = bli_zreal(ap[i__3]);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0. || (bli_zreal(y[i__3]) != 0. || 
			bli_zimag(y[i__3]) != 0.)) {
		    bla_d_cnjg(&z__2, &y[jy]);
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(z__2) - bli_zimag(*alpha) * bli_zimag(z__2)), (bli_zreal(*alpha) * bli_zimag(z__2) + bli_zimag(*alpha) * bli_zreal(z__2)), z__1 );
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp1 );
		    i__2 = jx;
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(x[i__2]) - bli_zimag(*alpha) * bli_zimag(x[i__2])), (bli_zreal(*alpha) * bli_zimag(x[i__2]) + bli_zimag(*alpha) * bli_zreal(x[i__2])), z__2 );
		    bla_d_cnjg(&z__1, &z__2);
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp2 );
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			bli_zsets( (bli_zreal(x[i__5]) * bli_zreal(temp1) - bli_zimag(x[i__5]) * bli_zimag(temp1)), (bli_zreal(x[i__5]) * bli_zimag(temp1) + bli_zimag(x[i__5]) * bli_zreal(temp1)), z__3 );
			bli_zsets( (bli_zreal(ap[i__4]) + bli_zreal(z__3)), (bli_zimag(ap[i__4]) + bli_zimag(z__3)), z__2 );
			i__6 = iy;
			bli_zsets( (bli_zreal(y[i__6]) * bli_zreal(temp2) - bli_zimag(y[i__6]) * bli_zimag(temp2)), (bli_zreal(y[i__6]) * bli_zimag(temp2) + bli_zimag(y[i__6]) * bli_zreal(temp2)), z__4 );
			bli_zsets( (bli_zreal(z__2) + bli_zreal(z__4)), (bli_zimag(z__2) + bli_zimag(z__4)), z__1 );
			bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), ap[i__3] );
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    bli_zsets( (bli_zreal(x[i__4]) * bli_zreal(temp1) - bli_zimag(x[i__4]) * bli_zimag(temp1)), (bli_zreal(x[i__4]) * bli_zimag(temp1) + bli_zimag(x[i__4]) * bli_zreal(temp1)), z__2 );
		    i__5 = jy;
		    bli_zsets( (bli_zreal(y[i__5]) * bli_zreal(temp2) - bli_zimag(y[i__5]) * bli_zimag(temp2)), (bli_zreal(y[i__5]) * bli_zimag(temp2) + bli_zimag(y[i__5]) * bli_zreal(temp2)), z__3 );
		    bli_zsets( (bli_zreal(z__2) + bli_zreal(z__3)), (bli_zimag(z__2) + bli_zimag(z__3)), z__1 );
		    d__1 = bli_zreal(ap[i__3]) + bli_zreal(z__1);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    d__1 = bli_zreal(ap[i__3]);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		}
		jx += *incx;
		jy += *incy;
		kk += j;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when lower triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0. || (bli_zreal(y[i__3]) != 0. || 
			bli_zimag(y[i__3]) != 0.)) {
		    bla_d_cnjg(&z__2, &y[j]);
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(z__2) - bli_zimag(*alpha) * bli_zimag(z__2)), (bli_zreal(*alpha) * bli_zimag(z__2) + bli_zimag(*alpha) * bli_zreal(z__2)), z__1 );
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp1 );
		    i__2 = j;
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(x[i__2]) - bli_zimag(*alpha) * bli_zimag(x[i__2])), (bli_zreal(*alpha) * bli_zimag(x[i__2]) + bli_zimag(*alpha) * bli_zreal(x[i__2])), z__2 );
		    bla_d_cnjg(&z__1, &z__2);
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp2 );
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    bli_zsets( (bli_zreal(x[i__4]) * bli_zreal(temp1) - bli_zimag(x[i__4]) * bli_zimag(temp1)), (bli_zreal(x[i__4]) * bli_zimag(temp1) + bli_zimag(x[i__4]) * bli_zreal(temp1)), z__2 );
		    i__5 = j;
		    bli_zsets( (bli_zreal(y[i__5]) * bli_zreal(temp2) - bli_zimag(y[i__5]) * bli_zimag(temp2)), (bli_zreal(y[i__5]) * bli_zimag(temp2) + bli_zimag(y[i__5]) * bli_zreal(temp2)), z__3 );
		    bli_zsets( (bli_zreal(z__2) + bli_zreal(z__3)), (bli_zimag(z__2) + bli_zimag(z__3)), z__1 );
		    d__1 = bli_zreal(ap[i__3]) + bli_zreal(z__1);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			bli_zsets( (bli_zreal(x[i__5]) * bli_zreal(temp1) - bli_zimag(x[i__5]) * bli_zimag(temp1)), (bli_zreal(x[i__5]) * bli_zimag(temp1) + bli_zimag(x[i__5]) * bli_zreal(temp1)), z__3 );
			bli_zsets( (bli_zreal(ap[i__4]) + bli_zreal(z__3)), (bli_zimag(ap[i__4]) + bli_zimag(z__3)), z__2 );
			i__6 = i__;
			bli_zsets( (bli_zreal(y[i__6]) * bli_zreal(temp2) - bli_zimag(y[i__6]) * bli_zimag(temp2)), (bli_zreal(y[i__6]) * bli_zimag(temp2) + bli_zimag(y[i__6]) * bli_zreal(temp2)), z__4 );
			bli_zsets( (bli_zreal(z__2) + bli_zreal(z__4)), (bli_zimag(z__2) + bli_zimag(z__4)), z__1 );
			bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), ap[i__3] );
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    d__1 = bli_zreal(ap[i__3]);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (bli_zreal(x[i__2]) != 0. || bli_zimag(x[i__2]) != 0. || (bli_zreal(y[i__3]) != 0. || 
			bli_zimag(y[i__3]) != 0.)) {
		    bla_d_cnjg(&z__2, &y[jy]);
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(z__2) - bli_zimag(*alpha) * bli_zimag(z__2)), (bli_zreal(*alpha) * bli_zimag(z__2) + bli_zimag(*alpha) * bli_zreal(z__2)), z__1 );
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp1 );
		    i__2 = jx;
		    bli_zsets( (bli_zreal(*alpha) * bli_zreal(x[i__2]) - bli_zimag(*alpha) * bli_zimag(x[i__2])), (bli_zreal(*alpha) * bli_zimag(x[i__2]) + bli_zimag(*alpha) * bli_zreal(x[i__2])), z__2 );
		    bla_d_cnjg(&z__1, &z__2);
		    bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), temp2 );
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    bli_zsets( (bli_zreal(x[i__4]) * bli_zreal(temp1) - bli_zimag(x[i__4]) * bli_zimag(temp1)), (bli_zreal(x[i__4]) * bli_zimag(temp1) + bli_zimag(x[i__4]) * bli_zreal(temp1)), z__2 );
		    i__5 = jy;
		    bli_zsets( (bli_zreal(y[i__5]) * bli_zreal(temp2) - bli_zimag(y[i__5]) * bli_zimag(temp2)), (bli_zreal(y[i__5]) * bli_zimag(temp2) + bli_zimag(y[i__5]) * bli_zreal(temp2)), z__3 );
		    bli_zsets( (bli_zreal(z__2) + bli_zreal(z__3)), (bli_zimag(z__2) + bli_zimag(z__3)), z__1 );
		    d__1 = bli_zreal(ap[i__3]) + bli_zreal(z__1);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			bli_zsets( (bli_zreal(x[i__5]) * bli_zreal(temp1) - bli_zimag(x[i__5]) * bli_zimag(temp1)), (bli_zreal(x[i__5]) * bli_zimag(temp1) + bli_zimag(x[i__5]) * bli_zreal(temp1)), z__3 );
			bli_zsets( (bli_zreal(ap[i__4]) + bli_zreal(z__3)), (bli_zimag(ap[i__4]) + bli_zimag(z__3)), z__2 );
			i__6 = iy;
			bli_zsets( (bli_zreal(y[i__6]) * bli_zreal(temp2) - bli_zimag(y[i__6]) * bli_zimag(temp2)), (bli_zreal(y[i__6]) * bli_zimag(temp2) + bli_zimag(y[i__6]) * bli_zreal(temp2)), z__4 );
			bli_zsets( (bli_zreal(z__2) + bli_zreal(z__4)), (bli_zimag(z__2) + bli_zimag(z__4)), z__1 );
			bli_zsets( (bli_zreal(z__1)), (bli_zimag(z__1)), ap[i__3] );
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    d__1 = bli_zreal(ap[i__3]);
		    bli_zsets( (d__1), (0.), ap[i__2] );
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return 0;

/*     End of ZHPR2 . */

} /* zhpr2_ */

#ifdef BLIS_ENABLE_BLAS

int PASTEF77(c,hpr2)(const bla_character *uplo, const bla_integer *n, const bla_scomplex *alpha, const bla_scomplex *x, const bla_integer *incx, const bla_scomplex *y, const bla_integer *incy, bla_scomplex *ap)
{
  return PASTEF77S(c,hpr2)( uplo, n, alpha, x, incx, y, incy, ap ); 
}

int PASTEF77(z,hpr2)(const bla_character *uplo, const bla_integer *n, const bla_dcomplex *alpha, const bla_dcomplex *x, const bla_integer *incx, const bla_dcomplex *y, const bla_integer *incy, bla_dcomplex *ap)
{
  return PASTEF77S(z,hpr2)( uplo, n, alpha, x, incx, y, incy, ap ); 
}

#endif

