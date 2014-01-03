/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#ifdef BLIS_ENABLE_BLAS2BLIS

/* chpr2.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,hpr2)(character *uplo, integer *n, singlecomplex *alpha, singlecomplex *x, integer *incx, singlecomplex *y, integer *incy, singlecomplex *ap)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1;
    singlecomplex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    void bla_r_cnjg(singlecomplex *, singlecomplex *);

    /* Local variables */
    integer info;
    singlecomplex temp1, temp2;
    integer i__, j, k;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kk, ix, iy, jx = 0, jy = 0, kx = 0, ky = 0;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);

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
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
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
	PASTEF770(xerbla)("CHPR2 ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->real == 0.f && alpha->imag == 0.f)) {
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
    if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].real != 0.f || x[i__2].imag != 0.f || (y[i__3].real != 0.f 
			|| y[i__3].imag != 0.f)) {
		    bla_r_cnjg(&q__2, &y[j]);
		    q__1.real = alpha->real * q__2.real - alpha->imag * q__2.imag, q__1.imag = 
			    alpha->real * q__2.imag + alpha->imag * q__2.real;
		    temp1.real = q__1.real, temp1.imag = q__1.imag;
		    i__2 = j;
		    q__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    q__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_r_cnjg(&q__1, &q__2);
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				q__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			q__2.real = ap[i__4].real + q__3.real, q__2.imag = ap[i__4].imag + 
				q__3.imag;
			i__6 = i__;
			q__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				q__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			q__1.real = q__2.real + q__4.real, q__1.imag = q__2.imag + q__4.imag;
			ap[i__3].real = q__1.real, ap[i__3].imag = q__1.imag;
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    q__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    q__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = j;
		    q__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    q__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    q__1.real = q__2.real + q__3.real, q__1.imag = q__2.imag + q__3.imag;
		    r__1 = ap[i__3].real + q__1.real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = ap[i__3].real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].real != 0.f || x[i__2].imag != 0.f || (y[i__3].real != 0.f 
			|| y[i__3].imag != 0.f)) {
		    bla_r_cnjg(&q__2, &y[jy]);
		    q__1.real = alpha->real * q__2.real - alpha->imag * q__2.imag, q__1.imag = 
			    alpha->real * q__2.imag + alpha->imag * q__2.real;
		    temp1.real = q__1.real, temp1.imag = q__1.imag;
		    i__2 = jx;
		    q__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    q__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_r_cnjg(&q__1, &q__2);
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				q__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			q__2.real = ap[i__4].real + q__3.real, q__2.imag = ap[i__4].imag + 
				q__3.imag;
			i__6 = iy;
			q__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				q__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			q__1.real = q__2.real + q__4.real, q__1.imag = q__2.imag + q__4.imag;
			ap[i__3].real = q__1.real, ap[i__3].imag = q__1.imag;
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    q__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    q__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = jy;
		    q__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    q__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    q__1.real = q__2.real + q__3.real, q__1.imag = q__2.imag + q__3.imag;
		    r__1 = ap[i__3].real + q__1.real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = ap[i__3].real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
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
		if (x[i__2].real != 0.f || x[i__2].imag != 0.f || (y[i__3].real != 0.f 
			|| y[i__3].imag != 0.f)) {
		    bla_r_cnjg(&q__2, &y[j]);
		    q__1.real = alpha->real * q__2.real - alpha->imag * q__2.imag, q__1.imag = 
			    alpha->real * q__2.imag + alpha->imag * q__2.real;
		    temp1.real = q__1.real, temp1.imag = q__1.imag;
		    i__2 = j;
		    q__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    q__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_r_cnjg(&q__1, &q__2);
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    q__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    q__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = j;
		    q__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    q__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    q__1.real = q__2.real + q__3.real, q__1.imag = q__2.imag + q__3.imag;
		    r__1 = ap[i__3].real + q__1.real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				q__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			q__2.real = ap[i__4].real + q__3.real, q__2.imag = ap[i__4].imag + 
				q__3.imag;
			i__6 = i__;
			q__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				q__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			q__1.real = q__2.real + q__4.real, q__1.imag = q__2.imag + q__4.imag;
			ap[i__3].real = q__1.real, ap[i__3].imag = q__1.imag;
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = ap[i__3].real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].real != 0.f || x[i__2].imag != 0.f || (y[i__3].real != 0.f 
			|| y[i__3].imag != 0.f)) {
		    bla_r_cnjg(&q__2, &y[jy]);
		    q__1.real = alpha->real * q__2.real - alpha->imag * q__2.imag, q__1.imag = 
			    alpha->real * q__2.imag + alpha->imag * q__2.real;
		    temp1.real = q__1.real, temp1.imag = q__1.imag;
		    i__2 = jx;
		    q__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    q__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_r_cnjg(&q__1, &q__2);
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    q__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    q__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = jy;
		    q__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    q__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    q__1.real = q__2.real + q__3.real, q__1.imag = q__2.imag + q__3.imag;
		    r__1 = ap[i__3].real + q__1.real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				q__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			q__2.real = ap[i__4].real + q__3.real, q__2.imag = ap[i__4].imag + 
				q__3.imag;
			i__6 = iy;
			q__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				q__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			q__1.real = q__2.real + q__4.real, q__1.imag = q__2.imag + q__4.imag;
			ap[i__3].real = q__1.real, ap[i__3].imag = q__1.imag;
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = ap[i__3].real;
		    ap[i__2].real = r__1, ap[i__2].imag = 0.f;
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of CHPR2 . */

} /* chpr2_ */

/* zhpr2.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,hpr2)(character *uplo, integer *n, doublecomplex *alpha, doublecomplex *x, integer *incx, doublecomplex *y, integer *incy, doublecomplex *ap)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    void bla_d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer info;
    doublecomplex temp1, temp2;
    integer i__, j, k;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kk, ix, iy, jx = 0, jy = 0, kx = 0, ky = 0;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);

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
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
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
	PASTEF770(xerbla)("ZHPR2 ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->real == 0. && alpha->imag == 0.)) {
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
    if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].real != 0. || x[i__2].imag != 0. || (y[i__3].real != 0. || 
			y[i__3].imag != 0.)) {
		    bla_d_cnjg(&z__2, &y[j]);
		    z__1.real = alpha->real * z__2.real - alpha->imag * z__2.imag, z__1.imag = 
			    alpha->real * z__2.imag + alpha->imag * z__2.real;
		    temp1.real = z__1.real, temp1.imag = z__1.imag;
		    i__2 = j;
		    z__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    z__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_d_cnjg(&z__1, &z__2);
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			z__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				z__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			z__2.real = ap[i__4].real + z__3.real, z__2.imag = ap[i__4].imag + 
				z__3.imag;
			i__6 = i__;
			z__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				z__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			z__1.real = z__2.real + z__4.real, z__1.imag = z__2.imag + z__4.imag;
			ap[i__3].real = z__1.real, ap[i__3].imag = z__1.imag;
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    z__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    z__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = j;
		    z__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    z__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    z__1.real = z__2.real + z__3.real, z__1.imag = z__2.imag + z__3.imag;
		    d__1 = ap[i__3].real + z__1.real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    d__1 = ap[i__3].real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].real != 0. || x[i__2].imag != 0. || (y[i__3].real != 0. || 
			y[i__3].imag != 0.)) {
		    bla_d_cnjg(&z__2, &y[jy]);
		    z__1.real = alpha->real * z__2.real - alpha->imag * z__2.imag, z__1.imag = 
			    alpha->real * z__2.imag + alpha->imag * z__2.real;
		    temp1.real = z__1.real, temp1.imag = z__1.imag;
		    i__2 = jx;
		    z__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    z__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_d_cnjg(&z__1, &z__2);
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			z__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				z__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			z__2.real = ap[i__4].real + z__3.real, z__2.imag = ap[i__4].imag + 
				z__3.imag;
			i__6 = iy;
			z__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				z__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			z__1.real = z__2.real + z__4.real, z__1.imag = z__2.imag + z__4.imag;
			ap[i__3].real = z__1.real, ap[i__3].imag = z__1.imag;
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    z__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    z__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = jy;
		    z__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    z__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    z__1.real = z__2.real + z__3.real, z__1.imag = z__2.imag + z__3.imag;
		    d__1 = ap[i__3].real + z__1.real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    d__1 = ap[i__3].real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
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
		if (x[i__2].real != 0. || x[i__2].imag != 0. || (y[i__3].real != 0. || 
			y[i__3].imag != 0.)) {
		    bla_d_cnjg(&z__2, &y[j]);
		    z__1.real = alpha->real * z__2.real - alpha->imag * z__2.imag, z__1.imag = 
			    alpha->real * z__2.imag + alpha->imag * z__2.real;
		    temp1.real = z__1.real, temp1.imag = z__1.imag;
		    i__2 = j;
		    z__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    z__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_d_cnjg(&z__1, &z__2);
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    z__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    z__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = j;
		    z__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    z__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    z__1.real = z__2.real + z__3.real, z__1.imag = z__2.imag + z__3.imag;
		    d__1 = ap[i__3].real + z__1.real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			z__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				z__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			z__2.real = ap[i__4].real + z__3.real, z__2.imag = ap[i__4].imag + 
				z__3.imag;
			i__6 = i__;
			z__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				z__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			z__1.real = z__2.real + z__4.real, z__1.imag = z__2.imag + z__4.imag;
			ap[i__3].real = z__1.real, ap[i__3].imag = z__1.imag;
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    d__1 = ap[i__3].real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].real != 0. || x[i__2].imag != 0. || (y[i__3].real != 0. || 
			y[i__3].imag != 0.)) {
		    bla_d_cnjg(&z__2, &y[jy]);
		    z__1.real = alpha->real * z__2.real - alpha->imag * z__2.imag, z__1.imag = 
			    alpha->real * z__2.imag + alpha->imag * z__2.real;
		    temp1.real = z__1.real, temp1.imag = z__1.imag;
		    i__2 = jx;
		    z__2.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, 
			    z__2.imag = alpha->real * x[i__2].imag + alpha->imag * x[i__2]
			    .real;
		    bla_d_cnjg(&z__1, &z__2);
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    z__2.real = x[i__4].real * temp1.real - x[i__4].imag * temp1.imag, 
			    z__2.imag = x[i__4].real * temp1.imag + x[i__4].imag * 
			    temp1.real;
		    i__5 = jy;
		    z__3.real = y[i__5].real * temp2.real - y[i__5].imag * temp2.imag, 
			    z__3.imag = y[i__5].real * temp2.imag + y[i__5].imag * 
			    temp2.real;
		    z__1.real = z__2.real + z__3.real, z__1.imag = z__2.imag + z__3.imag;
		    d__1 = ap[i__3].real + z__1.real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			z__3.real = x[i__5].real * temp1.real - x[i__5].imag * temp1.imag, 
				z__3.imag = x[i__5].real * temp1.imag + x[i__5].imag * 
				temp1.real;
			z__2.real = ap[i__4].real + z__3.real, z__2.imag = ap[i__4].imag + 
				z__3.imag;
			i__6 = iy;
			z__4.real = y[i__6].real * temp2.real - y[i__6].imag * temp2.imag, 
				z__4.imag = y[i__6].real * temp2.imag + y[i__6].imag * 
				temp2.real;
			z__1.real = z__2.real + z__4.real, z__1.imag = z__2.imag + z__4.imag;
			ap[i__3].real = z__1.real, ap[i__3].imag = z__1.imag;
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    d__1 = ap[i__3].real;
		    ap[i__2].real = d__1, ap[i__2].imag = 0.;
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of ZHPR2 . */

} /* zhpr2_ */

#endif

