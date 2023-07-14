/* f2c_slartg.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLARTG generates a plane rotation with bla_real cosine and bla_real sine. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLARTG + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slartg. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slartg. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slartg. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SLARTG( F, G, CS, SN, R ) */
 /* .. Scalar Arguments .. */
 /* REAL CS, F, G, R, SN */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLARTG generate a plane rotation so that */
 /* > */
 /* > [ CS SN ] . [ F ] = [ R ] where CS**2 + SN**2 = 1. */
 /* > [ -SN CS ] [ G ] [ 0 ] */
 /* > */
 /* > This is a slower, more accurate version of the BLAS1 routine SROTG, */
 /* > with the following other differences: */
 /* > F and G are unchanged on return. */
 /* > If G=0, then CS=1 and SN=0. */
 /* > If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any */
 /* > floating point operations (saves work in SBDSQR when */
 /* > there are zeros on the diagonal). */
 /* > */
 /* > If F exceeds G in magnitude, CS will be positive. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] F */
 /* > \verbatim */
 /* > F is REAL */
 /* > The first component of vector to be rotated. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] G */
 /* > \verbatim */
 /* > G is REAL */
 /* > The second component of vector to be rotated. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] CS */
 /* > \verbatim */
 /* > CS is REAL */
 /* > The cosine of the rotation. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] SN */
 /* > \verbatim */
 /* > SN is REAL */
 /* > The sine of the rotation. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] R */
 /* > \verbatim */
 /* > R is REAL */
 /* > The nonzero component of the rotated vector. */
 /* > */
 /* > This version has a few statements commented out for thread safety */
 /* > (machine parameters are computed on each entry). 10 feb 03, SJH. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* ===================================================================== */
 int f2c_slartg_(bla_real *f, bla_real *g, bla_real *cs, bla_real *sn, bla_real *r__) {
 /* System generated locals */
 bla_integer i__1;
 bla_real r__1, r__2;
 /* Builtin functions */
 /* Local variables */
 bla_integer i__;
 bla_real f1, g1, eps, scale;
 bla_integer count;
 bla_real safmn2, safmx2;
 bla_real safmin;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Parameters .. */
 /* .. */
 /* .. Local Scalars .. */
 /* LOGICAL FIRST */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Save statement .. */
 /* SAVE FIRST, SAFMX2, SAFMIN, SAFMN2 */
 /* .. */
 /* .. Data statements .. */
 /* DATA FIRST / .TRUE. / */
 /* .. */
 /* .. Executable Statements .. */
 /* IF( FIRST ) THEN */
 safmin = bla_slamch_("S", (ftnlen)1);
 eps = bla_slamch_("E", (ftnlen)1);
 r__1 = bla_slamch_("B", (ftnlen)1);
 i__1 = (bla_integer) (log(safmin / eps) / log(bla_slamch_("B", (ftnlen)1)) / 2.f);
 safmn2 = bla_pow_ri(&r__1, &i__1);
 safmx2 = 1.f / safmn2;
 /* FIRST = .FALSE. */
 /* END IF */
 if (*g == 0.f) {
 *cs = 1.f;
 *sn = 0.f;
 *r__ = *f;
 }
 else if (*f == 0.f) {
 *cs = 0.f;
 *sn = 1.f;
 *r__ = *g;
 }
 else {
 f1 = *f;
 g1 = *g;
 /* Computing MAX */
 r__1 = bla_r_abs(f1), r__2 = bla_r_abs(g1);
 scale = bla_a_max(r__1,r__2);
 if (scale >= safmx2) {
 count = 0;
 L10: ++count;
 f1 *= safmn2;
 g1 *= safmn2;
 /* Computing MAX */
 r__1 = bla_r_abs(f1), r__2 = bla_r_abs(g1);
 scale = bla_a_max(r__1,r__2);
 if (scale >= safmx2 && count < 20) {
 goto L10;
 }
 /* Computing 2nd power */
 r__1 = f1;
 /* Computing 2nd power */
 r__2 = g1;
 *r__ = sqrt(r__1 * r__1 + r__2 * r__2);
 *cs = f1 / *r__;
 *sn = g1 / *r__;
 i__1 = count;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 *r__ *= safmx2;
 /* L20: */
 }
 }
 else if (scale <= safmn2) {
 count = 0;
 L30: ++count;
 f1 *= safmx2;
 g1 *= safmx2;
 /* Computing MAX */
 r__1 = bla_r_abs(f1), r__2 = bla_r_abs(g1);
 scale = bla_a_max(r__1,r__2);
 if (scale <= safmn2) {
 goto L30;
 }
 /* Computing 2nd power */
 r__1 = f1;
 /* Computing 2nd power */
 r__2 = g1;
 *r__ = sqrt(r__1 * r__1 + r__2 * r__2);
 *cs = f1 / *r__;
 *sn = g1 / *r__;
 i__1 = count;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 *r__ *= safmn2;
 /* L40: */
 }
 }
 else {
 /* Computing 2nd power */
 r__1 = f1;
 /* Computing 2nd power */
 r__2 = g1;
 *r__ = sqrt(r__1 * r__1 + r__2 * r__2);
 *cs = f1 / *r__;
 *sn = g1 / *r__;
 }
 if (bla_r_abs(*f) > bla_r_abs(*g) && *cs < 0.f) {
 *cs = -(*cs);
 *sn = -(*sn);
 *r__ = -(*r__);
 }
 }
 return 0;
 /* End of SLARTG */
 }
 /* f2c_slartg_ */
 
