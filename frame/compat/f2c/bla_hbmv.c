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

/* chbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,hbmv)(character *uplo, integer *n, integer *k, singlecomplex * alpha, singlecomplex *a, integer *lda, singlecomplex *x, integer *incx, singlecomplex *beta, singlecomplex *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1;
    singlecomplex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    void bla_r_cnjg(singlecomplex *, singlecomplex *);

    /* Local variables */
    integer info;
    singlecomplex temp1, temp2;
    integer i__, j, l;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kplus1, ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHBMV  performs the matrix-vector  operation */

/*     y := alpha*A*x + beta*y, */

/*  where alpha and beta are scalars, x and y are n element vectors and */
/*  A is an n by n hermitian band matrix, with k super-diagonals. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the band matrix A is being supplied as */
/*           follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  being supplied. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  being supplied. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry, K specifies the number of super-diagonals of the */
/*           matrix A. K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX         . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX          array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the hermitian matrix, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer the upper */
/*           triangular part of a hermitian band matrix from conventional */
/*           full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the hermitian matrix, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer the lower */
/*           triangular part of a hermitian band matrix from conventional */
/*           full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set and are assumed to be zero. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - COMPLEX         . */
/*           On entry, BETA specifies the scalar beta. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX          array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*k < 0) {
	info = 3;
    } else if (*lda < *k + 1) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    } else if (*incy == 0) {
	info = 11;
    }
    if (info != 0) {
	PASTEF770(xerbla)("CHBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->real == 0.f && alpha->imag == 0.f && (beta->real == 1.f && 
	    beta->imag == 0.f))) {
	return 0;
    }

/*     Set up the start points in  X  and  Y. */

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

/*     Start the operations. In this version the elements of the array A */
/*     are accessed sequentially with one pass through A. */

/*     First form  y := beta*y. */

    if (beta->real != 1.f || beta->imag != 0.f) {
	if (*incy == 1) {
	    if (beta->real == 0.f && beta->imag == 0.f) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    y[i__2].real = 0.f, y[i__2].imag = 0.f;
/* L10: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    q__1.real = beta->real * y[i__3].real - beta->imag * y[i__3].imag, 
			    q__1.imag = beta->real * y[i__3].imag + beta->imag * y[i__3]
			    .real;
		    y[i__2].real = q__1.real, y[i__2].imag = q__1.imag;
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (beta->real == 0.f && beta->imag == 0.f) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    y[i__2].real = 0.f, y[i__2].imag = 0.f;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    i__3 = iy;
		    q__1.real = beta->real * y[i__3].real - beta->imag * y[i__3].imag, 
			    q__1.imag = beta->real * y[i__3].imag + beta->imag * y[i__3]
			    .real;
		    y[i__2].real = q__1.real, y[i__2].imag = q__1.imag;
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (alpha->real == 0.f && alpha->imag == 0.f) {
	return 0;
    }
    if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  y  when upper triangle of A is stored. */

	kplus1 = *k + 1;
	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		q__1.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, q__1.imag =
			 alpha->real * x[i__2].imag + alpha->imag * x[i__2].real;
		temp1.real = q__1.real, temp1.imag = q__1.imag;
		temp2.real = 0.f, temp2.imag = 0.f;
		l = kplus1 - j;
/* Computing MAX */
		i__2 = 1, i__3 = j - *k;
		i__4 = j - 1;
		for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    i__5 = l + i__ + j * a_dim1;
		    q__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    q__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    q__1.real = y[i__3].real + q__2.real, q__1.imag = y[i__3].imag + q__2.imag;
		    y[i__2].real = q__1.real, y[i__2].imag = q__1.imag;
		    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
		    i__2 = i__;
		    q__2.real = q__3.real * x[i__2].real - q__3.imag * x[i__2].imag, q__2.imag =
			     q__3.real * x[i__2].imag + q__3.imag * x[i__2].real;
		    q__1.real = temp2.real + q__2.real, q__1.imag = temp2.imag + q__2.imag;
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
/* L50: */
		}
		i__4 = j;
		i__2 = j;
		i__3 = kplus1 + j * a_dim1;
		r__1 = a[i__3].real;
		q__3.real = r__1 * temp1.real, q__3.imag = r__1 * temp1.imag;
		q__2.real = y[i__2].real + q__3.real, q__2.imag = y[i__2].imag + q__3.imag;
		q__4.real = alpha->real * temp2.real - alpha->imag * temp2.imag, q__4.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		q__1.real = q__2.real + q__4.real, q__1.imag = q__2.imag + q__4.imag;
		y[i__4].real = q__1.real, y[i__4].imag = q__1.imag;
/* L60: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__4 = jx;
		q__1.real = alpha->real * x[i__4].real - alpha->imag * x[i__4].imag, q__1.imag =
			 alpha->real * x[i__4].imag + alpha->imag * x[i__4].real;
		temp1.real = q__1.real, temp1.imag = q__1.imag;
		temp2.real = 0.f, temp2.imag = 0.f;
		ix = kx;
		iy = ky;
		l = kplus1 - j;
/* Computing MAX */
		i__4 = 1, i__2 = j - *k;
		i__3 = j - 1;
		for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
		    i__4 = iy;
		    i__2 = iy;
		    i__5 = l + i__ + j * a_dim1;
		    q__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    q__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    q__1.real = y[i__2].real + q__2.real, q__1.imag = y[i__2].imag + q__2.imag;
		    y[i__4].real = q__1.real, y[i__4].imag = q__1.imag;
		    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
		    i__4 = ix;
		    q__2.real = q__3.real * x[i__4].real - q__3.imag * x[i__4].imag, q__2.imag =
			     q__3.real * x[i__4].imag + q__3.imag * x[i__4].real;
		    q__1.real = temp2.real + q__2.real, q__1.imag = temp2.imag + q__2.imag;
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
		    ix += *incx;
		    iy += *incy;
/* L70: */
		}
		i__3 = jy;
		i__4 = jy;
		i__2 = kplus1 + j * a_dim1;
		r__1 = a[i__2].real;
		q__3.real = r__1 * temp1.real, q__3.imag = r__1 * temp1.imag;
		q__2.real = y[i__4].real + q__3.real, q__2.imag = y[i__4].imag + q__3.imag;
		q__4.real = alpha->real * temp2.real - alpha->imag * temp2.imag, q__4.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		q__1.real = q__2.real + q__4.real, q__1.imag = q__2.imag + q__4.imag;
		y[i__3].real = q__1.real, y[i__3].imag = q__1.imag;
		jx += *incx;
		jy += *incy;
		if (j > *k) {
		    kx += *incx;
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y  when lower triangle of A is stored. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__3 = j;
		q__1.real = alpha->real * x[i__3].real - alpha->imag * x[i__3].imag, q__1.imag =
			 alpha->real * x[i__3].imag + alpha->imag * x[i__3].real;
		temp1.real = q__1.real, temp1.imag = q__1.imag;
		temp2.real = 0.f, temp2.imag = 0.f;
		i__3 = j;
		i__4 = j;
		i__2 = j * a_dim1 + 1;
		r__1 = a[i__2].real;
		q__2.real = r__1 * temp1.real, q__2.imag = r__1 * temp1.imag;
		q__1.real = y[i__4].real + q__2.real, q__1.imag = y[i__4].imag + q__2.imag;
		y[i__3].real = q__1.real, y[i__3].imag = q__1.imag;
		l = 1 - j;
/* Computing MIN */
		i__4 = *n, i__2 = j + *k;
		i__3 = f2c_min(i__4,i__2);
		for (i__ = j + 1; i__ <= i__3; ++i__) {
		    i__4 = i__;
		    i__2 = i__;
		    i__5 = l + i__ + j * a_dim1;
		    q__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    q__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    q__1.real = y[i__2].real + q__2.real, q__1.imag = y[i__2].imag + q__2.imag;
		    y[i__4].real = q__1.real, y[i__4].imag = q__1.imag;
		    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
		    i__4 = i__;
		    q__2.real = q__3.real * x[i__4].real - q__3.imag * x[i__4].imag, q__2.imag =
			     q__3.real * x[i__4].imag + q__3.imag * x[i__4].real;
		    q__1.real = temp2.real + q__2.real, q__1.imag = temp2.imag + q__2.imag;
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
/* L90: */
		}
		i__3 = j;
		i__4 = j;
		q__2.real = alpha->real * temp2.real - alpha->imag * temp2.imag, q__2.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		q__1.real = y[i__4].real + q__2.real, q__1.imag = y[i__4].imag + q__2.imag;
		y[i__3].real = q__1.real, y[i__3].imag = q__1.imag;
/* L100: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__3 = jx;
		q__1.real = alpha->real * x[i__3].real - alpha->imag * x[i__3].imag, q__1.imag =
			 alpha->real * x[i__3].imag + alpha->imag * x[i__3].real;
		temp1.real = q__1.real, temp1.imag = q__1.imag;
		temp2.real = 0.f, temp2.imag = 0.f;
		i__3 = jy;
		i__4 = jy;
		i__2 = j * a_dim1 + 1;
		r__1 = a[i__2].real;
		q__2.real = r__1 * temp1.real, q__2.imag = r__1 * temp1.imag;
		q__1.real = y[i__4].real + q__2.real, q__1.imag = y[i__4].imag + q__2.imag;
		y[i__3].real = q__1.real, y[i__3].imag = q__1.imag;
		l = 1 - j;
		ix = jx;
		iy = jy;
/* Computing MIN */
		i__4 = *n, i__2 = j + *k;
		i__3 = f2c_min(i__4,i__2);
		for (i__ = j + 1; i__ <= i__3; ++i__) {
		    ix += *incx;
		    iy += *incy;
		    i__4 = iy;
		    i__2 = iy;
		    i__5 = l + i__ + j * a_dim1;
		    q__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    q__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    q__1.real = y[i__2].real + q__2.real, q__1.imag = y[i__2].imag + q__2.imag;
		    y[i__4].real = q__1.real, y[i__4].imag = q__1.imag;
		    bla_r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
		    i__4 = ix;
		    q__2.real = q__3.real * x[i__4].real - q__3.imag * x[i__4].imag, q__2.imag =
			     q__3.real * x[i__4].imag + q__3.imag * x[i__4].real;
		    q__1.real = temp2.real + q__2.real, q__1.imag = temp2.imag + q__2.imag;
		    temp2.real = q__1.real, temp2.imag = q__1.imag;
/* L110: */
		}
		i__3 = jy;
		i__4 = jy;
		q__2.real = alpha->real * temp2.real - alpha->imag * temp2.imag, q__2.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		q__1.real = y[i__4].real + q__2.real, q__1.imag = y[i__4].imag + q__2.imag;
		y[i__3].real = q__1.real, y[i__3].imag = q__1.imag;
		jx += *incx;
		jy += *incy;
/* L120: */
	    }
	}
    }

    return 0;

/*     End of CHBMV . */

} /* chbmv_ */

/* zhbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,hbmv)(character *uplo, integer *n, integer *k, doublecomplex *alpha, doublecomplex *a, integer *lda, doublecomplex *x, integer * incx, doublecomplex *beta, doublecomplex *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    void bla_d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer info;
    doublecomplex temp1, temp2;
    integer i__, j, l;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kplus1, ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZHBMV  performs the matrix-vector  operation */

/*     y := alpha*A*x + beta*y, */

/*  where alpha and beta are scalars, x and y are n element vectors and */
/*  A is an n by n hermitian band matrix, with k super-diagonals. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the band matrix A is being supplied as */
/*           follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  being supplied. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  being supplied. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry, K specifies the number of super-diagonals of the */
/*           matrix A. K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX*16      . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the hermitian matrix, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer the upper */
/*           triangular part of a hermitian band matrix from conventional */
/*           full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the hermitian matrix, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer the lower */
/*           triangular part of a hermitian band matrix from conventional */
/*           full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set and are assumed to be zero. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - COMPLEX*16      . */
/*           On entry, BETA specifies the scalar beta. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX*16       array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*k < 0) {
	info = 3;
    } else if (*lda < *k + 1) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    } else if (*incy == 0) {
	info = 11;
    }
    if (info != 0) {
	PASTEF770(xerbla)("ZHBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->real == 0. && alpha->imag == 0. && (beta->real == 1. && 
	    beta->imag == 0.))) {
	return 0;
    }

/*     Set up the start points in  X  and  Y. */

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

/*     Start the operations. In this version the elements of the array A */
/*     are accessed sequentially with one pass through A. */

/*     First form  y := beta*y. */

    if (beta->real != 1. || beta->imag != 0.) {
	if (*incy == 1) {
	    if (beta->real == 0. && beta->imag == 0.) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    y[i__2].real = 0., y[i__2].imag = 0.;
/* L10: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    z__1.real = beta->real * y[i__3].real - beta->imag * y[i__3].imag, 
			    z__1.imag = beta->real * y[i__3].imag + beta->imag * y[i__3]
			    .real;
		    y[i__2].real = z__1.real, y[i__2].imag = z__1.imag;
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (beta->real == 0. && beta->imag == 0.) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    y[i__2].real = 0., y[i__2].imag = 0.;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    i__3 = iy;
		    z__1.real = beta->real * y[i__3].real - beta->imag * y[i__3].imag, 
			    z__1.imag = beta->real * y[i__3].imag + beta->imag * y[i__3]
			    .real;
		    y[i__2].real = z__1.real, y[i__2].imag = z__1.imag;
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (alpha->real == 0. && alpha->imag == 0.) {
	return 0;
    }
    if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  y  when upper triangle of A is stored. */

	kplus1 = *k + 1;
	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		z__1.real = alpha->real * x[i__2].real - alpha->imag * x[i__2].imag, z__1.imag =
			 alpha->real * x[i__2].imag + alpha->imag * x[i__2].real;
		temp1.real = z__1.real, temp1.imag = z__1.imag;
		temp2.real = 0., temp2.imag = 0.;
		l = kplus1 - j;
/* Computing MAX */
		i__2 = 1, i__3 = j - *k;
		i__4 = j - 1;
		for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    i__5 = l + i__ + j * a_dim1;
		    z__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    z__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    z__1.real = y[i__3].real + z__2.real, z__1.imag = y[i__3].imag + z__2.imag;
		    y[i__2].real = z__1.real, y[i__2].imag = z__1.imag;
		    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
		    i__2 = i__;
		    z__2.real = z__3.real * x[i__2].real - z__3.imag * x[i__2].imag, z__2.imag =
			     z__3.real * x[i__2].imag + z__3.imag * x[i__2].real;
		    z__1.real = temp2.real + z__2.real, z__1.imag = temp2.imag + z__2.imag;
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
/* L50: */
		}
		i__4 = j;
		i__2 = j;
		i__3 = kplus1 + j * a_dim1;
		d__1 = a[i__3].real;
		z__3.real = d__1 * temp1.real, z__3.imag = d__1 * temp1.imag;
		z__2.real = y[i__2].real + z__3.real, z__2.imag = y[i__2].imag + z__3.imag;
		z__4.real = alpha->real * temp2.real - alpha->imag * temp2.imag, z__4.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		z__1.real = z__2.real + z__4.real, z__1.imag = z__2.imag + z__4.imag;
		y[i__4].real = z__1.real, y[i__4].imag = z__1.imag;
/* L60: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__4 = jx;
		z__1.real = alpha->real * x[i__4].real - alpha->imag * x[i__4].imag, z__1.imag =
			 alpha->real * x[i__4].imag + alpha->imag * x[i__4].real;
		temp1.real = z__1.real, temp1.imag = z__1.imag;
		temp2.real = 0., temp2.imag = 0.;
		ix = kx;
		iy = ky;
		l = kplus1 - j;
/* Computing MAX */
		i__4 = 1, i__2 = j - *k;
		i__3 = j - 1;
		for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
		    i__4 = iy;
		    i__2 = iy;
		    i__5 = l + i__ + j * a_dim1;
		    z__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    z__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    z__1.real = y[i__2].real + z__2.real, z__1.imag = y[i__2].imag + z__2.imag;
		    y[i__4].real = z__1.real, y[i__4].imag = z__1.imag;
		    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
		    i__4 = ix;
		    z__2.real = z__3.real * x[i__4].real - z__3.imag * x[i__4].imag, z__2.imag =
			     z__3.real * x[i__4].imag + z__3.imag * x[i__4].real;
		    z__1.real = temp2.real + z__2.real, z__1.imag = temp2.imag + z__2.imag;
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
		    ix += *incx;
		    iy += *incy;
/* L70: */
		}
		i__3 = jy;
		i__4 = jy;
		i__2 = kplus1 + j * a_dim1;
		d__1 = a[i__2].real;
		z__3.real = d__1 * temp1.real, z__3.imag = d__1 * temp1.imag;
		z__2.real = y[i__4].real + z__3.real, z__2.imag = y[i__4].imag + z__3.imag;
		z__4.real = alpha->real * temp2.real - alpha->imag * temp2.imag, z__4.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		z__1.real = z__2.real + z__4.real, z__1.imag = z__2.imag + z__4.imag;
		y[i__3].real = z__1.real, y[i__3].imag = z__1.imag;
		jx += *incx;
		jy += *incy;
		if (j > *k) {
		    kx += *incx;
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y  when lower triangle of A is stored. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__3 = j;
		z__1.real = alpha->real * x[i__3].real - alpha->imag * x[i__3].imag, z__1.imag =
			 alpha->real * x[i__3].imag + alpha->imag * x[i__3].real;
		temp1.real = z__1.real, temp1.imag = z__1.imag;
		temp2.real = 0., temp2.imag = 0.;
		i__3 = j;
		i__4 = j;
		i__2 = j * a_dim1 + 1;
		d__1 = a[i__2].real;
		z__2.real = d__1 * temp1.real, z__2.imag = d__1 * temp1.imag;
		z__1.real = y[i__4].real + z__2.real, z__1.imag = y[i__4].imag + z__2.imag;
		y[i__3].real = z__1.real, y[i__3].imag = z__1.imag;
		l = 1 - j;
/* Computing MIN */
		i__4 = *n, i__2 = j + *k;
		i__3 = f2c_min(i__4,i__2);
		for (i__ = j + 1; i__ <= i__3; ++i__) {
		    i__4 = i__;
		    i__2 = i__;
		    i__5 = l + i__ + j * a_dim1;
		    z__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    z__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    z__1.real = y[i__2].real + z__2.real, z__1.imag = y[i__2].imag + z__2.imag;
		    y[i__4].real = z__1.real, y[i__4].imag = z__1.imag;
		    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
		    i__4 = i__;
		    z__2.real = z__3.real * x[i__4].real - z__3.imag * x[i__4].imag, z__2.imag =
			     z__3.real * x[i__4].imag + z__3.imag * x[i__4].real;
		    z__1.real = temp2.real + z__2.real, z__1.imag = temp2.imag + z__2.imag;
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
/* L90: */
		}
		i__3 = j;
		i__4 = j;
		z__2.real = alpha->real * temp2.real - alpha->imag * temp2.imag, z__2.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		z__1.real = y[i__4].real + z__2.real, z__1.imag = y[i__4].imag + z__2.imag;
		y[i__3].real = z__1.real, y[i__3].imag = z__1.imag;
/* L100: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__3 = jx;
		z__1.real = alpha->real * x[i__3].real - alpha->imag * x[i__3].imag, z__1.imag =
			 alpha->real * x[i__3].imag + alpha->imag * x[i__3].real;
		temp1.real = z__1.real, temp1.imag = z__1.imag;
		temp2.real = 0., temp2.imag = 0.;
		i__3 = jy;
		i__4 = jy;
		i__2 = j * a_dim1 + 1;
		d__1 = a[i__2].real;
		z__2.real = d__1 * temp1.real, z__2.imag = d__1 * temp1.imag;
		z__1.real = y[i__4].real + z__2.real, z__1.imag = y[i__4].imag + z__2.imag;
		y[i__3].real = z__1.real, y[i__3].imag = z__1.imag;
		l = 1 - j;
		ix = jx;
		iy = jy;
/* Computing MIN */
		i__4 = *n, i__2 = j + *k;
		i__3 = f2c_min(i__4,i__2);
		for (i__ = j + 1; i__ <= i__3; ++i__) {
		    ix += *incx;
		    iy += *incy;
		    i__4 = iy;
		    i__2 = iy;
		    i__5 = l + i__ + j * a_dim1;
		    z__2.real = temp1.real * a[i__5].real - temp1.imag * a[i__5].imag, 
			    z__2.imag = temp1.real * a[i__5].imag + temp1.imag * a[i__5]
			    .real;
		    z__1.real = y[i__2].real + z__2.real, z__1.imag = y[i__2].imag + z__2.imag;
		    y[i__4].real = z__1.real, y[i__4].imag = z__1.imag;
		    bla_d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
		    i__4 = ix;
		    z__2.real = z__3.real * x[i__4].real - z__3.imag * x[i__4].imag, z__2.imag =
			     z__3.real * x[i__4].imag + z__3.imag * x[i__4].real;
		    z__1.real = temp2.real + z__2.real, z__1.imag = temp2.imag + z__2.imag;
		    temp2.real = z__1.real, temp2.imag = z__1.imag;
/* L110: */
		}
		i__3 = jy;
		i__4 = jy;
		z__2.real = alpha->real * temp2.real - alpha->imag * temp2.imag, z__2.imag = 
			alpha->real * temp2.imag + alpha->imag * temp2.real;
		z__1.real = y[i__4].real + z__2.real, z__1.imag = y[i__4].imag + z__2.imag;
		y[i__3].real = z__1.real, y[i__3].imag = z__1.imag;
		jx += *incx;
		jy += *incy;
/* L120: */
	    }
	}
    }

    return 0;

/*     End of ZHBMV . */

} /* zhbmv_ */

#endif

