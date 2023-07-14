/* f2c_zlassq.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b ZLASSQ updates a sum of squares represented in scaled form. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download ZLASSQ + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlassq. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlassq. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlassq. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE ZLASSQ( N, X, INCX, SCALE, SUMSQ ) */
 /* .. Scalar Arguments .. */
 /* INTEGER INCX, N */
 /* DOUBLE PRECISION SCALE, SUMSQ */
 /* .. */
 /* .. Array Arguments .. */
 /* COMPLEX*16 X( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > ZLASSQ returns the values scl and ssq such that */
 /* > */
 /* > ( scl**2 )*ssq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq, */
 /* > */
 /* > where x( i ) = bla_d_abs( X( 1 + ( i - 1 )*INCX ) ). The value of sumsq is */
 /* > assumed to be at least unity and the value of ssq will then satisfy */
 /* > */
 /* > 1.0 <= ssq <= ( sumsq + 2*n ). */
 /* > */
 /* > scale is assumed to be non-negative and scl returns the value */
 /* > */
 /* > scl = bla_a_max( scale, bla_d_abs( bla_d_real( x( i ) ) ), bla_d_abs( bla_d_imag( x( i ) ) ) ), */
 /* > i */
 /* > */
 /* > scale and sumsq must be supplied in SCALE and SUMSQ respectively. */
 /* > SCALE and SUMSQ are overwritten by scl and ssq respectively. */
 /* > */
 /* > The routine makes only one pass through the vector X. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The number of elements to be used from the vector X. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] X */
 /* > \verbatim */
 /* > X is COMPLEX*16 array, dimension (1+(N-1)*INCX) */
 /* > The vector x as described above. */
 /* > x( i ) = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] INCX */
 /* > \verbatim */
 /* > INCX is INTEGER */
 /* > The increment between successive values of the vector X. */
 /* > INCX > 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] SCALE */
 /* > \verbatim */
 /* > SCALE is DOUBLE PRECISION */
 /* > On entry, the value scale in the equation above. */
 /* > On exit, SCALE is overwritten with the value scl . */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] SUMSQ */
 /* > \verbatim */
 /* > SUMSQ is DOUBLE PRECISION */
 /* > On entry, the value sumsq in the equation above. */
 /* > On exit, SUMSQ is overwritten with the value ssq . */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplex16OTHERauxiliary */
 /* ===================================================================== */
 int f2c_zlassq_(bla_integer *n, bla_dcomplex *x, bla_integer *incx, bla_double *scale, bla_double *sumsq) {
 /* System generated locals */
 bla_integer i__1, i__2, i__3;
 bla_double d__1;
 /* Builtin functions */
 /* Local variables */
 bla_integer ix;
 bla_double temp1;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* .. Array Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Parameters .. */
 /* .. */
 /* .. Local Scalars .. */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Parameter adjustments */
 --x;
 /* Function Body */
 if (*n > 0) {
 i__1 = (*n - 1) * *incx + 1;
 i__2 = *incx;
 for (ix = 1;
 i__2 < 0 ? ix >= i__1 : ix <= i__1;
 ix += i__2) {
 i__3 = ix;
 temp1 = (d__1 = x[i__3].real, bla_d_abs(d__1));
 if (temp1 > 0. || f2c_disnan_(&temp1)) {
 if (*scale < temp1) {
 /* Computing 2nd power */
 d__1 = *scale / temp1;
 *sumsq = *sumsq * (d__1 * d__1) + 1;
 *scale = temp1;
 }
 else {
 /* Computing 2nd power */
 d__1 = temp1 / *scale;
 *sumsq += d__1 * d__1;
 }
 }
 temp1 = (d__1 = bla_d_imag(&x[ix]), bla_d_abs(d__1));
 if (temp1 > 0. || f2c_disnan_(&temp1)) {
 if (*scale < temp1) {
 /* Computing 2nd power */
 d__1 = *scale / temp1;
 *sumsq = *sumsq * (d__1 * d__1) + 1;
 *scale = temp1;
 }
 else {
 /* Computing 2nd power */
 d__1 = temp1 / *scale;
 *sumsq += d__1 * d__1;
 }
 }
 /* L10: */
 }
 }
 return 0;
 /* End of ZLASSQ */
 }
 /* f2c_zlassq_ */
 
