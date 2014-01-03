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

/* ctpsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,tpsv)(character *uplo, character *trans, character *diag, integer *n, singlecomplex *ap, singlecomplex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5;
    singlecomplex q__1, q__2, q__3;

    /* Builtin functions */
    void bla_c_div(singlecomplex *, singlecomplex *, singlecomplex *), bla_r_cnjg(singlecomplex *, singlecomplex *);

    /* Local variables */
    integer info;
    singlecomplex temp;
    integer i__, j, k;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kk, ix, jx, kx = 0;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);
    logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTPSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular matrix, supplied in packed form. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   conjg( A' )*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX          array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
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
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF770(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF770(xerbla)("CTPSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = PASTEF770(lsame)(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = PASTEF770(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (x[i__1].real != 0.f || x[i__1].imag != 0.f) {
			if (nounit) {
			    i__1 = j;
			    bla_c_div(&q__1, &x[j], &ap[kk]);
			    x[i__1].real = q__1.real, x[i__1].imag = q__1.imag;
			}
			i__1 = j;
			temp.real = x[i__1].real, temp.imag = x[i__1].imag;
			k = kk - 1;
			for (i__ = j - 1; i__ >= 1; --i__) {
			    i__1 = i__;
			    i__2 = i__;
			    i__3 = k;
			    q__2.real = temp.real * ap[i__3].real - temp.imag * ap[i__3]
				    .imag, q__2.imag = temp.real * ap[i__3].imag + temp.imag 
				    * ap[i__3].real;
			    q__1.real = x[i__2].real - q__2.real, q__1.imag = x[i__2].imag - 
				    q__2.imag;
			    x[i__1].real = q__1.real, x[i__1].imag = q__1.imag;
			    --k;
/* L10: */
			}
		    }
		    kk -= j;
/* L20: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    if (x[i__1].real != 0.f || x[i__1].imag != 0.f) {
			if (nounit) {
			    i__1 = jx;
			    bla_c_div(&q__1, &x[jx], &ap[kk]);
			    x[i__1].real = q__1.real, x[i__1].imag = q__1.imag;
			}
			i__1 = jx;
			temp.real = x[i__1].real, temp.imag = x[i__1].imag;
			ix = jx;
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = k;
			    q__2.real = temp.real * ap[i__4].real - temp.imag * ap[i__4]
				    .imag, q__2.imag = temp.real * ap[i__4].imag + temp.imag 
				    * ap[i__4].real;
			    q__1.real = x[i__3].real - q__2.real, q__1.imag = x[i__3].imag - 
				    q__2.imag;
			    x[i__2].real = q__1.real, x[i__2].imag = q__1.imag;
/* L30: */
			}
		    }
		    jx -= *incx;
		    kk -= j;
/* L40: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (x[i__2].real != 0.f || x[i__2].imag != 0.f) {
			if (nounit) {
			    i__2 = j;
			    bla_c_div(&q__1, &x[j], &ap[kk]);
			    x[i__2].real = q__1.real, x[i__2].imag = q__1.imag;
			}
			i__2 = j;
			temp.real = x[i__2].real, temp.imag = x[i__2].imag;
			k = kk + 1;
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = k;
			    q__2.real = temp.real * ap[i__5].real - temp.imag * ap[i__5]
				    .imag, q__2.imag = temp.real * ap[i__5].imag + temp.imag 
				    * ap[i__5].real;
			    q__1.real = x[i__4].real - q__2.real, q__1.imag = x[i__4].imag - 
				    q__2.imag;
			    x[i__3].real = q__1.real, x[i__3].imag = q__1.imag;
			    ++k;
/* L50: */
			}
		    }
		    kk += *n - j + 1;
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    if (x[i__2].real != 0.f || x[i__2].imag != 0.f) {
			if (nounit) {
			    i__2 = jx;
			    bla_c_div(&q__1, &x[jx], &ap[kk]);
			    x[i__2].real = q__1.real, x[i__2].imag = q__1.imag;
			}
			i__2 = jx;
			temp.real = x[i__2].real, temp.imag = x[i__2].imag;
			ix = jx;
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = k;
			    q__2.real = temp.real * ap[i__5].real - temp.imag * ap[i__5]
				    .imag, q__2.imag = temp.real * ap[i__5].imag + temp.imag 
				    * ap[i__5].real;
			    q__1.real = x[i__4].real - q__2.real, q__1.imag = x[i__4].imag - 
				    q__2.imag;
			    x[i__3].real = q__1.real, x[i__3].imag = q__1.imag;
/* L70: */
			}
		    }
		    jx += *incx;
		    kk += *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x  or  x := inv( conjg( A' ) )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    temp.real = x[i__2].real, temp.imag = x[i__2].imag;
		    k = kk;
		    if (noconj) {
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = k;
			    i__4 = i__;
			    q__2.real = ap[i__3].real * x[i__4].real - ap[i__3].imag * x[
				    i__4].imag, q__2.imag = ap[i__3].real * x[i__4].imag 
				    + ap[i__3].imag * x[i__4].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    ++k;
/* L90: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &ap[kk + j - 1]);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    } else {
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__3 = i__;
			    q__2.real = q__3.real * x[i__3].real - q__3.imag * x[i__3].imag, 
				    q__2.imag = q__3.real * x[i__3].imag + q__3.imag * x[
				    i__3].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    ++k;
/* L100: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk + j - 1]);
			    bla_c_div(&q__1, &temp, &q__2);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    }
		    i__2 = j;
		    x[i__2].real = temp.real, x[i__2].imag = temp.imag;
		    kk += j;
/* L110: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    temp.real = x[i__2].real, temp.imag = x[i__2].imag;
		    ix = kx;
		    if (noconj) {
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    i__3 = k;
			    i__4 = ix;
			    q__2.real = ap[i__3].real * x[i__4].real - ap[i__3].imag * x[
				    i__4].imag, q__2.imag = ap[i__3].real * x[i__4].imag 
				    + ap[i__3].imag * x[i__4].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    ix += *incx;
/* L120: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &ap[kk + j - 1]);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    } else {
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__3 = ix;
			    q__2.real = q__3.real * x[i__3].real - q__3.imag * x[i__3].imag, 
				    q__2.imag = q__3.real * x[i__3].imag + q__3.imag * x[
				    i__3].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    ix += *incx;
/* L130: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk + j - 1]);
			    bla_c_div(&q__1, &temp, &q__2);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    }
		    i__2 = jx;
		    x[i__2].real = temp.real, x[i__2].imag = temp.imag;
		    jx += *incx;
		    kk += j;
/* L140: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    temp.real = x[i__1].real, temp.imag = x[i__1].imag;
		    k = kk;
		    if (noconj) {
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = k;
			    i__3 = i__;
			    q__2.real = ap[i__2].real * x[i__3].real - ap[i__2].imag * x[
				    i__3].imag, q__2.imag = ap[i__2].real * x[i__3].imag 
				    + ap[i__2].imag * x[i__3].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    --k;
/* L150: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &ap[kk - *n + j]);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    } else {
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__2 = i__;
			    q__2.real = q__3.real * x[i__2].real - q__3.imag * x[i__2].imag, 
				    q__2.imag = q__3.real * x[i__2].imag + q__3.imag * x[
				    i__2].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    --k;
/* L160: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk - *n + j]);
			    bla_c_div(&q__1, &temp, &q__2);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    }
		    i__1 = j;
		    x[i__1].real = temp.real, x[i__1].imag = temp.imag;
		    kk -= *n - j + 1;
/* L170: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    temp.real = x[i__1].real, temp.imag = x[i__1].imag;
		    ix = kx;
		    if (noconj) {
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    i__2 = k;
			    i__3 = ix;
			    q__2.real = ap[i__2].real * x[i__3].real - ap[i__2].imag * x[
				    i__3].imag, q__2.imag = ap[i__2].real * x[i__3].imag 
				    + ap[i__2].imag * x[i__3].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    ix -= *incx;
/* L180: */
			}
			if (nounit) {
			    bla_c_div(&q__1, &temp, &ap[kk - *n + j]);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    } else {
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    bla_r_cnjg(&q__3, &ap[k]);
			    i__2 = ix;
			    q__2.real = q__3.real * x[i__2].real - q__3.imag * x[i__2].imag, 
				    q__2.imag = q__3.real * x[i__2].imag + q__3.imag * x[
				    i__2].real;
			    q__1.real = temp.real - q__2.real, q__1.imag = temp.imag - 
				    q__2.imag;
			    temp.real = q__1.real, temp.imag = q__1.imag;
			    ix -= *incx;
/* L190: */
			}
			if (nounit) {
			    bla_r_cnjg(&q__2, &ap[kk - *n + j]);
			    bla_c_div(&q__1, &temp, &q__2);
			    temp.real = q__1.real, temp.imag = q__1.imag;
			}
		    }
		    i__1 = jx;
		    x[i__1].real = temp.real, x[i__1].imag = temp.imag;
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of CTPSV . */

} /* ctpsv_ */

/* dtpsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,tpsv)(character *uplo, character *trans, character *diag, integer *n, doublereal *ap, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    integer info;
    doublereal temp;
    integer i__, j, k;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kk, ix, jx, kx = 0;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);
    logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTPSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular matrix, supplied in packed form. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   A'*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - DOUBLE PRECISION array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
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
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF770(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF770(xerbla)("DTPSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = PASTEF770(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.) {
			if (nounit) {
			    x[j] /= ap[kk];
			}
			temp = x[j];
			k = kk - 1;
			for (i__ = j - 1; i__ >= 1; --i__) {
			    x[i__] -= temp * ap[k];
			    --k;
/* L10: */
			}
		    }
		    kk -= j;
/* L20: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.) {
			if (nounit) {
			    x[jx] /= ap[kk];
			}
			temp = x[jx];
			ix = jx;
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    x[ix] -= temp * ap[k];
/* L30: */
			}
		    }
		    jx -= *incx;
		    kk -= j;
/* L40: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.) {
			if (nounit) {
			    x[j] /= ap[kk];
			}
			temp = x[j];
			k = kk + 1;
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[i__] -= temp * ap[k];
			    ++k;
/* L50: */
			}
		    }
		    kk += *n - j + 1;
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.) {
			if (nounit) {
			    x[jx] /= ap[kk];
			}
			temp = x[jx];
			ix = jx;
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    x[ix] -= temp * ap[k];
/* L70: */
			}
		    }
		    jx += *incx;
		    kk += *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp -= ap[k] * x[i__];
			++k;
/* L90: */
		    }
		    if (nounit) {
			temp /= ap[kk + j - 1];
		    }
		    x[j] = temp;
		    kk += j;
/* L100: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = kx;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			temp -= ap[k] * x[ix];
			ix += *incx;
/* L110: */
		    }
		    if (nounit) {
			temp /= ap[kk + j - 1];
		    }
		    x[jx] = temp;
		    jx += *incx;
		    kk += j;
/* L120: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    k = kk;
		    i__1 = j + 1;
		    for (i__ = *n; i__ >= i__1; --i__) {
			temp -= ap[k] * x[i__];
			--k;
/* L130: */
		    }
		    if (nounit) {
			temp /= ap[kk - *n + j];
		    }
		    x[j] = temp;
		    kk -= *n - j + 1;
/* L140: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = kx;
		    i__1 = kk - (*n - (j + 1));
		    for (k = kk; k >= i__1; --k) {
			temp -= ap[k] * x[ix];
			ix -= *incx;
/* L150: */
		    }
		    if (nounit) {
			temp /= ap[kk - *n + j];
		    }
		    x[jx] = temp;
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of DTPSV . */

} /* dtpsv_ */

/* stpsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,tpsv)(character *uplo, character *trans, character *diag, integer *n, real *ap, real *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    integer info;
    real temp;
    integer i__, j, k;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kk, ix, jx, kx = 0;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);
    logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  STPSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular matrix, supplied in packed form. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   A'*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - REAL             array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - REAL             array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
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
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF770(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF770(xerbla)("STPSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = PASTEF770(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.f) {
			if (nounit) {
			    x[j] /= ap[kk];
			}
			temp = x[j];
			k = kk - 1;
			for (i__ = j - 1; i__ >= 1; --i__) {
			    x[i__] -= temp * ap[k];
			    --k;
/* L10: */
			}
		    }
		    kk -= j;
/* L20: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.f) {
			if (nounit) {
			    x[jx] /= ap[kk];
			}
			temp = x[jx];
			ix = jx;
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    x[ix] -= temp * ap[k];
/* L30: */
			}
		    }
		    jx -= *incx;
		    kk -= j;
/* L40: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.f) {
			if (nounit) {
			    x[j] /= ap[kk];
			}
			temp = x[j];
			k = kk + 1;
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[i__] -= temp * ap[k];
			    ++k;
/* L50: */
			}
		    }
		    kk += *n - j + 1;
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.f) {
			if (nounit) {
			    x[jx] /= ap[kk];
			}
			temp = x[jx];
			ix = jx;
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    x[ix] -= temp * ap[k];
/* L70: */
			}
		    }
		    jx += *incx;
		    kk += *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp -= ap[k] * x[i__];
			++k;
/* L90: */
		    }
		    if (nounit) {
			temp /= ap[kk + j - 1];
		    }
		    x[j] = temp;
		    kk += j;
/* L100: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = kx;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			temp -= ap[k] * x[ix];
			ix += *incx;
/* L110: */
		    }
		    if (nounit) {
			temp /= ap[kk + j - 1];
		    }
		    x[jx] = temp;
		    jx += *incx;
		    kk += j;
/* L120: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    k = kk;
		    i__1 = j + 1;
		    for (i__ = *n; i__ >= i__1; --i__) {
			temp -= ap[k] * x[i__];
			--k;
/* L130: */
		    }
		    if (nounit) {
			temp /= ap[kk - *n + j];
		    }
		    x[j] = temp;
		    kk -= *n - j + 1;
/* L140: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = kx;
		    i__1 = kk - (*n - (j + 1));
		    for (k = kk; k >= i__1; --k) {
			temp -= ap[k] * x[ix];
			ix -= *incx;
/* L150: */
		    }
		    if (nounit) {
			temp /= ap[kk - *n + j];
		    }
		    x[jx] = temp;
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of STPSV . */

} /* stpsv_ */

/* ztpsv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,tpsv)(character *uplo, character *trans, character *diag, integer *n, doublecomplex *ap, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    void bla_z_div(doublecomplex *, doublecomplex *, doublecomplex *), bla_d_cnjg(
	    doublecomplex *, doublecomplex *);

    /* Local variables */
    integer info;
    doublecomplex temp;
    integer i__, j, k;
    extern logical PASTEF770(lsame)(character *, character *, ftnlen, ftnlen);
    integer kk, ix, jx, kx = 0;
    extern /* Subroutine */ int PASTEF770(xerbla)(character *, integer *, ftnlen);
    logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTPSV  solves one of the systems of equations */

/*     A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b, */

/*  where b and x are n element vectors and A is an n by n unit, or */
/*  non-unit, upper or lower triangular matrix, supplied in packed form. */

/*  No test for singularity or near-singularity is included in this */
/*  routine. Such tests must be performed before calling this routine. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the equations to be solved as */
/*           follows: */

/*              TRANS = 'N' or 'n'   A*x = b. */

/*              TRANS = 'T' or 't'   A'*x = b. */

/*              TRANS = 'C' or 'c'   conjg( A' )*x = b. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX*16       array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element right-hand side vector b. On exit, X is overwritten */
/*           with the solution vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
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
    --x;
    --ap;

    /* Function Body */
    info = 0;
    if (! PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! PASTEF770(lsame)(diag, "U", (ftnlen)1, (ftnlen)1) && ! PASTEF770(lsame)(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	PASTEF770(xerbla)("ZTPSV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = PASTEF770(lsame)(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = PASTEF770(lsame)(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (PASTEF770(lsame)(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  x := inv( A )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (x[i__1].real != 0. || x[i__1].imag != 0.) {
			if (nounit) {
			    i__1 = j;
			    bla_z_div(&z__1, &x[j], &ap[kk]);
			    x[i__1].real = z__1.real, x[i__1].imag = z__1.imag;
			}
			i__1 = j;
			temp.real = x[i__1].real, temp.imag = x[i__1].imag;
			k = kk - 1;
			for (i__ = j - 1; i__ >= 1; --i__) {
			    i__1 = i__;
			    i__2 = i__;
			    i__3 = k;
			    z__2.real = temp.real * ap[i__3].real - temp.imag * ap[i__3]
				    .imag, z__2.imag = temp.real * ap[i__3].imag + temp.imag 
				    * ap[i__3].real;
			    z__1.real = x[i__2].real - z__2.real, z__1.imag = x[i__2].imag - 
				    z__2.imag;
			    x[i__1].real = z__1.real, x[i__1].imag = z__1.imag;
			    --k;
/* L10: */
			}
		    }
		    kk -= j;
/* L20: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    if (x[i__1].real != 0. || x[i__1].imag != 0.) {
			if (nounit) {
			    i__1 = jx;
			    bla_z_div(&z__1, &x[jx], &ap[kk]);
			    x[i__1].real = z__1.real, x[i__1].imag = z__1.imag;
			}
			i__1 = jx;
			temp.real = x[i__1].real, temp.imag = x[i__1].imag;
			ix = jx;
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = k;
			    z__2.real = temp.real * ap[i__4].real - temp.imag * ap[i__4]
				    .imag, z__2.imag = temp.real * ap[i__4].imag + temp.imag 
				    * ap[i__4].real;
			    z__1.real = x[i__3].real - z__2.real, z__1.imag = x[i__3].imag - 
				    z__2.imag;
			    x[i__2].real = z__1.real, x[i__2].imag = z__1.imag;
/* L30: */
			}
		    }
		    jx -= *incx;
		    kk -= j;
/* L40: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (x[i__2].real != 0. || x[i__2].imag != 0.) {
			if (nounit) {
			    i__2 = j;
			    bla_z_div(&z__1, &x[j], &ap[kk]);
			    x[i__2].real = z__1.real, x[i__2].imag = z__1.imag;
			}
			i__2 = j;
			temp.real = x[i__2].real, temp.imag = x[i__2].imag;
			k = kk + 1;
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = k;
			    z__2.real = temp.real * ap[i__5].real - temp.imag * ap[i__5]
				    .imag, z__2.imag = temp.real * ap[i__5].imag + temp.imag 
				    * ap[i__5].real;
			    z__1.real = x[i__4].real - z__2.real, z__1.imag = x[i__4].imag - 
				    z__2.imag;
			    x[i__3].real = z__1.real, x[i__3].imag = z__1.imag;
			    ++k;
/* L50: */
			}
		    }
		    kk += *n - j + 1;
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    if (x[i__2].real != 0. || x[i__2].imag != 0.) {
			if (nounit) {
			    i__2 = jx;
			    bla_z_div(&z__1, &x[jx], &ap[kk]);
			    x[i__2].real = z__1.real, x[i__2].imag = z__1.imag;
			}
			i__2 = jx;
			temp.real = x[i__2].real, temp.imag = x[i__2].imag;
			ix = jx;
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = k;
			    z__2.real = temp.real * ap[i__5].real - temp.imag * ap[i__5]
				    .imag, z__2.imag = temp.real * ap[i__5].imag + temp.imag 
				    * ap[i__5].real;
			    z__1.real = x[i__4].real - z__2.real, z__1.imag = x[i__4].imag - 
				    z__2.imag;
			    x[i__3].real = z__1.real, x[i__3].imag = z__1.imag;
/* L70: */
			}
		    }
		    jx += *incx;
		    kk += *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x  or  x := inv( conjg( A' ) )*x. */

	if (PASTEF770(lsame)(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    temp.real = x[i__2].real, temp.imag = x[i__2].imag;
		    k = kk;
		    if (noconj) {
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = k;
			    i__4 = i__;
			    z__2.real = ap[i__3].real * x[i__4].real - ap[i__3].imag * x[
				    i__4].imag, z__2.imag = ap[i__3].real * x[i__4].imag 
				    + ap[i__3].imag * x[i__4].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    ++k;
/* L90: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &ap[kk + j - 1]);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    } else {
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__3 = i__;
			    z__2.real = z__3.real * x[i__3].real - z__3.imag * x[i__3].imag, 
				    z__2.imag = z__3.real * x[i__3].imag + z__3.imag * x[
				    i__3].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    ++k;
/* L100: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk + j - 1]);
			    bla_z_div(&z__1, &temp, &z__2);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    }
		    i__2 = j;
		    x[i__2].real = temp.real, x[i__2].imag = temp.imag;
		    kk += j;
/* L110: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    temp.real = x[i__2].real, temp.imag = x[i__2].imag;
		    ix = kx;
		    if (noconj) {
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    i__3 = k;
			    i__4 = ix;
			    z__2.real = ap[i__3].real * x[i__4].real - ap[i__3].imag * x[
				    i__4].imag, z__2.imag = ap[i__3].real * x[i__4].imag 
				    + ap[i__3].imag * x[i__4].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    ix += *incx;
/* L120: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &ap[kk + j - 1]);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    } else {
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__3 = ix;
			    z__2.real = z__3.real * x[i__3].real - z__3.imag * x[i__3].imag, 
				    z__2.imag = z__3.real * x[i__3].imag + z__3.imag * x[
				    i__3].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    ix += *incx;
/* L130: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk + j - 1]);
			    bla_z_div(&z__1, &temp, &z__2);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    }
		    i__2 = jx;
		    x[i__2].real = temp.real, x[i__2].imag = temp.imag;
		    jx += *incx;
		    kk += j;
/* L140: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    temp.real = x[i__1].real, temp.imag = x[i__1].imag;
		    k = kk;
		    if (noconj) {
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = k;
			    i__3 = i__;
			    z__2.real = ap[i__2].real * x[i__3].real - ap[i__2].imag * x[
				    i__3].imag, z__2.imag = ap[i__2].real * x[i__3].imag 
				    + ap[i__2].imag * x[i__3].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    --k;
/* L150: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &ap[kk - *n + j]);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    } else {
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__2 = i__;
			    z__2.real = z__3.real * x[i__2].real - z__3.imag * x[i__2].imag, 
				    z__2.imag = z__3.real * x[i__2].imag + z__3.imag * x[
				    i__2].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    --k;
/* L160: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk - *n + j]);
			    bla_z_div(&z__1, &temp, &z__2);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    }
		    i__1 = j;
		    x[i__1].real = temp.real, x[i__1].imag = temp.imag;
		    kk -= *n - j + 1;
/* L170: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    temp.real = x[i__1].real, temp.imag = x[i__1].imag;
		    ix = kx;
		    if (noconj) {
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    i__2 = k;
			    i__3 = ix;
			    z__2.real = ap[i__2].real * x[i__3].real - ap[i__2].imag * x[
				    i__3].imag, z__2.imag = ap[i__2].real * x[i__3].imag 
				    + ap[i__2].imag * x[i__3].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    ix -= *incx;
/* L180: */
			}
			if (nounit) {
			    bla_z_div(&z__1, &temp, &ap[kk - *n + j]);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    } else {
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    bla_d_cnjg(&z__3, &ap[k]);
			    i__2 = ix;
			    z__2.real = z__3.real * x[i__2].real - z__3.imag * x[i__2].imag, 
				    z__2.imag = z__3.real * x[i__2].imag + z__3.imag * x[
				    i__2].real;
			    z__1.real = temp.real - z__2.real, z__1.imag = temp.imag - 
				    z__2.imag;
			    temp.real = z__1.real, temp.imag = z__1.imag;
			    ix -= *incx;
/* L190: */
			}
			if (nounit) {
			    bla_d_cnjg(&z__2, &ap[kk - *n + j]);
			    bla_z_div(&z__1, &temp, &z__2);
			    temp.real = z__1.real, temp.imag = z__1.imag;
			}
		    }
		    i__1 = jx;
		    x[i__1].real = temp.real, x[i__1].imag = temp.imag;
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of ZTPSV . */

} /* ztpsv_ */

#endif

