/* f2c_slassq.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLASSQ updates a sum of squares represented in scaled form. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLASSQ + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slassq. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slassq. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slassq. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SLASSQ( N, X, INCX, SCALE, SUMSQ ) */
 /* .. Scalar Arguments .. */
 /* INTEGER INCX, N */
 /* REAL SCALE, SUMSQ */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL X( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLASSQ returns the values scl and smsq such that */
 /* > */
 /* > ( scl**2 )*smsq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq, */
 /* > */
 /* > where x( i ) = X( 1 + ( i - 1 )*INCX ). The value of sumsq is */
 /* > assumed to be non-negative and scl returns the value */
 /* > */
 /* > scl = bla_a_max( scale, bla_r_abs( x( i ) ) ). */
 /* > */
 /* > scale and sumsq must be supplied in SCALE and SUMSQ and */
 /* > scl and smsq are overwritten on SCALE and SUMSQ respectively. */
 /* > */
 /* > The routine makes only one pass through the vector x. */
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
 /* > X is REAL array, dimension (1+(N-1)*INCX) */
 /* > The vector for which a scaled sum of squares is computed. */
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
 /* > SCALE is REAL */
 /* > On entry, the value scale in the equation above. */
 /* > On exit, SCALE is overwritten with scl , the scaling factor */
 /* > for the sum of squares. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] SUMSQ */
 /* > \verbatim */
 /* > SUMSQ is REAL */
 /* > On entry, the value sumsq in the equation above. */
 /* > On exit, SUMSQ is overwritten with smsq , the basic sum of */
 /* > squares from which scl has been factored out. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* ===================================================================== */
 int f2c_slassq_(bla_integer *n, bla_real *x, bla_integer *incx, bla_real *scale, bla_real *sumsq) {
 /* System generated locals */
 bla_integer i__1, i__2;
 bla_real r__1;
 /* Local variables */
 bla_integer ix;
 bla_real absxi;
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
 absxi = (r__1 = x[ix], bla_r_abs(r__1));
 if (absxi > 0.f || f2c_sisnan_(&absxi)) {
 if (*scale < absxi) {
 /* Computing 2nd power */
 r__1 = *scale / absxi;
 *sumsq = *sumsq * (r__1 * r__1) + 1;
 *scale = absxi;
 }
 else {
 /* Computing 2nd power */
 r__1 = absxi / *scale;
 *sumsq += r__1 * r__1;
 }
 }
 /* L10: */
 }
 }
 return 0;
 /* End of SLASSQ */
 }
 /* f2c_slassq_ */
 
