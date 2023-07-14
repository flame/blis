/* f2c_zlarfg.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_dcomplex c_b5 = {
1.,0.}
;
 /* > \brief \b ZLARFG generates an elementary reflector (Householder matrix). */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download ZLARFG + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlarfg. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlarfg. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlarfg. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE ZLARFG( N, ALPHA, X, INCX, TAU ) */
 /* .. Scalar Arguments .. */
 /* INTEGER INCX, N */
 /* COMPLEX*16 ALPHA, TAU */
 /* .. */
 /* .. Array Arguments .. */
 /* COMPLEX*16 X( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > ZLARFG generates a bla_scomplex elementary reflector H of order n, such */
 /* > that */
 /* > */
 /* > H**H * ( alpha ) = ( beta ), H**H * H = I. */
 /* > ( x ) ( 0 ) */
 /* > */
 /* > where alpha and beta are scalars, with beta real, and x is an */
 /* > (n-1)-element bla_scomplex vector. H is represented in the form */
 /* > */
 /* > H = I - tau * ( 1 ) * ( 1 v**H ) , */
 /* > ( v ) */
 /* > */
 /* > where tau is a bla_scomplex scalar and v is a bla_scomplex (n-1)-element */
 /* > vector. Note that H is not hermitian. */
 /* > */
 /* > If the elements of x are all zero and alpha is real, then tau = 0 */
 /* > and H is taken to be the unit matrix. */
 /* > */
 /* > Otherwise 1 <= bla_d_real(tau) <= 2 and bla_d_abs(tau-1) <= 1 . */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the elementary reflector. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] ALPHA */
 /* > \verbatim */
 /* > ALPHA is COMPLEX*16 */
 /* > On entry, the value alpha. */
 /* > On exit, it is overwritten with the value beta. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] X */
 /* > \verbatim */
 /* > X is COMPLEX*16 array, dimension */
 /* > (1+(N-2)*abs(INCX)) */
 /* > On entry, the vector x. */
 /* > On exit, it is overwritten with the vector v. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] INCX */
 /* > \verbatim */
 /* > INCX is INTEGER */
 /* > The increment between elements of X. INCX > 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] TAU */
 /* > \verbatim */
 /* > TAU is COMPLEX*16 */
 /* > The value tau. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplex16OTHERauxiliary */
 /* ===================================================================== */
 int f2c_zlarfg_(bla_integer *n, bla_dcomplex *alpha, bla_dcomplex * x, bla_integer *incx, bla_dcomplex *tau) {
 /* System generated locals */
 bla_integer i__1;
 bla_double d__1, d__2;
 bla_dcomplex z__1, z__2;
 /* Builtin functions */
 /* Local variables */
 bla_integer j, knt;
 bla_double beta, alphi, alphr;
 bla_double xnorm;
 bla_double safmin;
 bla_double rsafmn;
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
 /* .. External Subroutines .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Parameter adjustments */
 --x;
 /* Function Body */
 if (*n <= 0) {
 tau->real = 0., tau->imag = 0.;
 return 0;
 }
 i__1 = *n - 1;
 xnorm = dznrm2_(&i__1, &x[1], incx);
 alphr = alpha->real;
 alphi = bla_d_imag(alpha);
 if (xnorm == 0. && alphi == 0.) {
 /* H = I */
 tau->real = 0., tau->imag = 0.;
 }
 else {
 /* general case */
 d__1 = f2c_dlapy3_(&alphr, &alphi, &xnorm);
 beta = -bla_d_sign(&d__1, &alphr);
 safmin = bla_dlamch_("S", (ftnlen)1) / bla_dlamch_("E", (ftnlen)1);
 rsafmn = 1. / safmin;
 knt = 0;
 if (bla_d_abs(beta) < safmin) {
 /* XNORM, BETA may be inaccurate;
 scale X and recompute them */
 L10: ++knt;
 i__1 = *n - 1;
 zdscal_(&i__1, &rsafmn, &x[1], incx);
 beta *= rsafmn;
 alphi *= rsafmn;
 alphr *= rsafmn;
 if (bla_d_abs(beta) < safmin && knt < 20) {
 goto L10;
 }
 /* New BETA is at most 1, at least SAFMIN */
 i__1 = *n - 1;
 xnorm = dznrm2_(&i__1, &x[1], incx);
 z__1.real = alphr, z__1.imag = alphi;
 alpha->real = z__1.real, alpha->imag = z__1.imag;
 d__1 = f2c_dlapy3_(&alphr, &alphi, &xnorm);
 beta = -bla_d_sign(&d__1, &alphr);
 }
 d__1 = (beta - alphr) / beta;
 d__2 = -alphi / beta;
 z__1.real = d__1, z__1.imag = d__2;
 tau->real = z__1.real, tau->imag = z__1.imag;
 z__2.real = alpha->real - beta, z__2.imag = alpha->imag;
 f2c_zladiv_(&z__1, &c_b5, &z__2);
 alpha->real = z__1.real, alpha->imag = z__1.imag;
 i__1 = *n - 1;
 zscal_(&i__1, alpha, &x[1], incx);
 /* If ALPHA is subnormal, it may lose relative accuracy */
 i__1 = knt;
 for (j = 1;
 j <= i__1;
 ++j) {
 beta *= safmin;
 /* L20: */
 }
 alpha->real = beta, alpha->imag = 0.;
 }
 return 0;
 /* End of ZLARFG */
 }
 /* f2c_zlarfg_ */
 
