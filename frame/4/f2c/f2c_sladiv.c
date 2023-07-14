/* f2c_sladiv.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLADIV performs bla_scomplex division in bla_real arithmetic, avoiding unnecessary overflow. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLADIV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sladiv. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sladiv. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sladiv. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SLADIV( A, B, C, D, P, Q ) */
 /* .. Scalar Arguments .. */
 /* REAL A, B, C, D, P, Q */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLADIV performs bla_scomplex division in bla_real arithmetic */
 /* > */
 /* > a + i*b */
 /* > p + i*q = --------- */
 /* > c + i*d */
 /* > */
 /* > The algorithm is due to Michael Baudin and Robert L. Smith */
 /* > and can be found in the paper */
 /* > "A Robust Complex Division in Scilab" */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] A */
 /* > \verbatim */
 /* > A is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[in] B */
 /* > \verbatim */
 /* > B is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[in] C */
 /* > \verbatim */
 /* > C is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[in] D */
 /* > \verbatim */
 /* > D is REAL */
 /* > The scalars a, b, c, and d in the above expression. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] P */
 /* > \verbatim */
 /* > P is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[out] Q */
 /* > \verbatim */
 /* > Q is REAL */
 /* > The scalars p and q in the above expression. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup realOTHERauxiliary */
 /* ===================================================================== */
 int f2c_sladiv_(bla_real *a, bla_real *b, bla_real *c__, bla_real *d__, bla_real *p, bla_real *q) {
 /* System generated locals */
 bla_real r__1, r__2;
 /* Local variables */
 bla_real s, aa, ab, bb, cc, cd, dd, be, un, ov, eps;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Parameters .. */
 /* .. Local Scalars .. */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. External Subroutines .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 aa = *a;
 bb = *b;
 cc = *c__;
 dd = *d__;
 /* Computing MAX */
 r__1 = bla_r_abs(*a), r__2 = bla_r_abs(*b);
 ab = bla_a_max(r__1,r__2);
 /* Computing MAX */
 r__1 = bla_r_abs(*c__), r__2 = bla_r_abs(*d__);
 cd = bla_a_max(r__1,r__2);
 s = 1.f;
 ov = bla_slamch_("Overflow threshold", (ftnlen)18);
 un = bla_slamch_("Safe minimum", (ftnlen)12);
 eps = bla_slamch_("Epsilon", (ftnlen)7);
 be = 2.f / (eps * eps);
 if (ab >= ov * .5f) {
 aa *= .5f;
 bb *= .5f;
 s *= 2.f;
 }
 if (cd >= ov * .5f) {
 cc *= .5f;
 dd *= .5f;
 s *= .5f;
 }
 if (ab <= un * 2.f / eps) {
 aa *= be;
 bb *= be;
 s /= be;
 }
 if (cd <= un * 2.f / eps) {
 cc *= be;
 dd *= be;
 s *= be;
 }
 if (bla_r_abs(*d__) <= bla_r_abs(*c__)) {
 f2c_sladiv1_(&aa, &bb, &cc, &dd, p, q);
 }
 else {
 f2c_sladiv1_(&bb, &aa, &dd, &cc, p, q);
 *q = -(*q);
 }
 *p *= s;
 *q *= s;
 return 0;
 /* End of SLADIV */
 }
 /* f2c_sladiv_ */
 /* > \ingroup realOTHERauxiliary */
 int f2c_sladiv1_(bla_real *a, bla_real *b, bla_real *c__, bla_real *d__, bla_real *p, bla_real *q) {
 bla_real r__, t;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Parameters .. */
 /* .. Local Scalars .. */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 r__ = *d__ / *c__;
 t = 1.f / (*c__ + *d__ * r__);
 *p = f2c_sladiv2_(a, b, c__, d__, &r__, &t);
 *a = -(*a);
 *q = f2c_sladiv2_(b, a, c__, d__, &r__, &t);
 return 0;
 /* End of SLADIV1 */
 }
 /* f2c_sladiv1_ */
 /* > \ingroup realOTHERauxiliary */
 bla_real f2c_sladiv2_(bla_real *a, bla_real *b, bla_real *c__, bla_real *d__, bla_real *r__, bla_real *t) {
 /* System generated locals */
 bla_real ret_val;
 /* Local variables */
 bla_real br;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Parameters .. */
 /* .. Local Scalars .. */
 /* .. */
 /* .. Executable Statements .. */
 if (*r__ != 0.f) {
 br = *b * *r__;
 if (br != 0.f) {
 ret_val = (*a + br) * *t;
 }
 else {
 ret_val = *a * *t + *b * *t * *r__;
 }
 }
 else {
 ret_val = (*a + *d__ * (*b / *c__)) * *t;
 }
 return ret_val;
 /* End of SLADIV */
 }
 /* f2c_sladiv2_ */
 
