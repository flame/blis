/* f2c_clacgv.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b CLACGV conjugates a bla_scomplex vector. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download CLACGV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/clacgv. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/clacgv. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/clacgv. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE CLACGV( N, X, INCX ) */
 /* .. Scalar Arguments .. */
 /* INTEGER INCX, N */
 /* .. */
 /* .. Array Arguments .. */
 /* COMPLEX X( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > CLACGV conjugates a bla_scomplex vector of length N. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The length of the vector X. N >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] X */
 /* > \verbatim */
 /* > X is COMPLEX array, dimension */
 /* > (1+(N-1)*abs(INCX)) */
 /* > On entry, the vector of length N to be conjugated. */
 /* > On exit, X is overwritten with conjg(X). */
 /* > \endverbatim */
 /* > */
 /* > \param[in] INCX */
 /* > \verbatim */
 /* > INCX is INTEGER */
 /* > The spacing between successive elements of X. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplexOTHERauxiliary */
 /* ===================================================================== */
 int f2c_clacgv_(bla_integer *n, bla_scomplex *x, bla_integer *incx) {
 /* System generated locals */
 bla_integer i__1, i__2;
 bla_scomplex q__1;
 /* Builtin functions */
 /* Local variables */
 bla_integer i__, ioff;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* .. Array Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Local Scalars .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Parameter adjustments */
 --x;
 /* Function Body */
 if (*incx == 1) {
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 i__2 = i__;
 bla_r_cnjg(&q__1, &x[i__]);
 x[i__2].real = q__1.real, x[i__2].imag = q__1.imag;
 /* L10: */
 }
 }
 else {
 ioff = 1;
 if (*incx < 0) {
 ioff = 1 - (*n - 1) * *incx;
 }
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 i__2 = ioff;
 bla_r_cnjg(&q__1, &x[ioff]);
 x[i__2].real = q__1.real, x[i__2].imag = q__1.imag;
 ioff += *incx;
 /* L20: */
 }
 }
 return 0;
 /* End of CLACGV */
 }
 /* f2c_clacgv_ */
 
