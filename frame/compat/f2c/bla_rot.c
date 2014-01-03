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

/* srot.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,rot)(integer *n, real *sx, integer *incx, real *sy, integer *incy, real *c__, real *s)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    real stemp;
    integer ix, iy;


/*     applies a plane rotation. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


    /* Parameter adjustments */
    --sy;
    --sx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       code for unequal increments or equal increments not equal */
/*         to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	stemp = *c__ * sx[ix] + *s * sy[iy];
	sy[iy] = *c__ * sy[iy] - *s * sx[ix];
	sx[ix] = stemp;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	stemp = *c__ * sx[i__] + *s * sy[i__];
	sy[i__] = *c__ * sy[i__] - *s * sx[i__];
	sx[i__] = stemp;
/* L30: */
    }
    return 0;
} /* srot_ */

/* drot.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,rot)(integer *n, doublereal *dx, integer *incx, doublereal *dy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    doublereal dtemp;
    integer ix, iy;


/*     applies a plane rotation. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       code for unequal increments or equal increments not equal */
/*         to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = *c__ * dx[ix] + *s * dy[iy];
	dy[iy] = *c__ * dy[iy] - *s * dx[ix];
	dx[ix] = dtemp;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = *c__ * dx[i__] + *s * dy[i__];
	dy[i__] = *c__ * dy[i__] - *s * dx[i__];
	dx[i__] = dtemp;
/* L30: */
    }
    return 0;
} /* drot_ */

/* csrot.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(cs,rot)(integer *n, singlecomplex *cx, integer *incx, singlecomplex *cy, integer *incy, real *c__, real *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    singlecomplex q__1, q__2, q__3;

    /* Local variables */
    integer i__;
    singlecomplex ctemp;
    integer ix, iy;


/*     applies a plane rotation, where the cos and sin (c and s) are real */
/*     and the vectors cx and cy are complex. */
/*     jack dongarra, linpack, 3/11/78. */


    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       code for unequal increments or equal increments not equal */
/*         to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	q__2.real = *c__ * cx[i__2].real, q__2.imag = *c__ * cx[i__2].imag;
	i__3 = iy;
	q__3.real = *s * cy[i__3].real, q__3.imag = *s * cy[i__3].imag;
	q__1.real = q__2.real + q__3.real, q__1.imag = q__2.imag + q__3.imag;
	ctemp.real = q__1.real, ctemp.imag = q__1.imag;
	i__2 = iy;
	i__3 = iy;
	q__2.real = *c__ * cy[i__3].real, q__2.imag = *c__ * cy[i__3].imag;
	i__4 = ix;
	q__3.real = *s * cx[i__4].real, q__3.imag = *s * cx[i__4].imag;
	q__1.real = q__2.real - q__3.real, q__1.imag = q__2.imag - q__3.imag;
	cy[i__2].real = q__1.real, cy[i__2].imag = q__1.imag;
	i__2 = ix;
	cx[i__2].real = ctemp.real, cx[i__2].imag = ctemp.imag;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	q__2.real = *c__ * cx[i__2].real, q__2.imag = *c__ * cx[i__2].imag;
	i__3 = i__;
	q__3.real = *s * cy[i__3].real, q__3.imag = *s * cy[i__3].imag;
	q__1.real = q__2.real + q__3.real, q__1.imag = q__2.imag + q__3.imag;
	ctemp.real = q__1.real, ctemp.imag = q__1.imag;
	i__2 = i__;
	i__3 = i__;
	q__2.real = *c__ * cy[i__3].real, q__2.imag = *c__ * cy[i__3].imag;
	i__4 = i__;
	q__3.real = *s * cx[i__4].real, q__3.imag = *s * cx[i__4].imag;
	q__1.real = q__2.real - q__3.real, q__1.imag = q__2.imag - q__3.imag;
	cy[i__2].real = q__1.real, cy[i__2].imag = q__1.imag;
	i__2 = i__;
	cx[i__2].real = ctemp.real, cx[i__2].imag = ctemp.imag;
/* L30: */
    }
    return 0;
} /* csrot_ */

/* zdrot.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(zd,rot)(integer *n, doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    integer i__;
    doublecomplex ztemp;
    integer ix, iy;


/*     applies a plane rotation, where the cos and sin (c and s) are */
/*     double precision and the vectors zx and zy are double complex. */
/*     jack dongarra, linpack, 3/11/78. */


    /* Parameter adjustments */
    --zy;
    --zx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*       code for unequal increments or equal increments not equal */
/*         to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	z__2.real = *c__ * zx[i__2].real, z__2.imag = *c__ * zx[i__2].imag;
	i__3 = iy;
	z__3.real = *s * zy[i__3].real, z__3.imag = *s * zy[i__3].imag;
	z__1.real = z__2.real + z__3.real, z__1.imag = z__2.imag + z__3.imag;
	ztemp.real = z__1.real, ztemp.imag = z__1.imag;
	i__2 = iy;
	i__3 = iy;
	z__2.real = *c__ * zy[i__3].real, z__2.imag = *c__ * zy[i__3].imag;
	i__4 = ix;
	z__3.real = *s * zx[i__4].real, z__3.imag = *s * zx[i__4].imag;
	z__1.real = z__2.real - z__3.real, z__1.imag = z__2.imag - z__3.imag;
	zy[i__2].real = z__1.real, zy[i__2].imag = z__1.imag;
	i__2 = ix;
	zx[i__2].real = ztemp.real, zx[i__2].imag = ztemp.imag;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	z__2.real = *c__ * zx[i__2].real, z__2.imag = *c__ * zx[i__2].imag;
	i__3 = i__;
	z__3.real = *s * zy[i__3].real, z__3.imag = *s * zy[i__3].imag;
	z__1.real = z__2.real + z__3.real, z__1.imag = z__2.imag + z__3.imag;
	ztemp.real = z__1.real, ztemp.imag = z__1.imag;
	i__2 = i__;
	i__3 = i__;
	z__2.real = *c__ * zy[i__3].real, z__2.imag = *c__ * zy[i__3].imag;
	i__4 = i__;
	z__3.real = *s * zx[i__4].real, z__3.imag = *s * zx[i__4].imag;
	z__1.real = z__2.real - z__3.real, z__1.imag = z__2.imag - z__3.imag;
	zy[i__2].real = z__1.real, zy[i__2].imag = z__1.imag;
	i__2 = i__;
	zx[i__2].real = ztemp.real, zx[i__2].imag = ztemp.imag;
/* L30: */
    }
    return 0;
} /* zdrot_ */

#endif

