/* f2c_dladiv.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b DLADIV performs bla_scomplex division in bla_real arithmetic, avoiding unnecessary overflow. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download DLADIV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dladiv. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dladiv. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dladiv. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE DLADIV( A, B, C, D, P, Q ) */
 /* .. Scalar Arguments .. */
 /* DOUBLE PRECISION A, B, C, D, P, Q */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > DLADIV performs bla_scomplex division in bla_real arithmetic */
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
 /* > A is DOUBLE PRECISION */
 /* > \endverbatim */
 /* > */
 /* > \param[in] B */
 /* > \verbatim */
 /* > B is DOUBLE PRECISION */
 /* > \endverbatim */
 /* > */
 /* > \param[in] C */
 /* > \verbatim */
 /* > C is DOUBLE PRECISION */
 /* > \endverbatim */
 /* > */
 /* > \param[in] D */
 /* > \verbatim */
 /* > D is DOUBLE PRECISION */
 /* > The scalars a, b, c, and d in the above expression. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] P */
 /* > \verbatim */
 /* > P is DOUBLE PRECISION */
 /* > \endverbatim */
 /* > */
 /* > \param[out] Q */
 /* > \verbatim */
 /* > Q is DOUBLE PRECISION */
 /* > The scalars p and q in the above expression. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup doubleOTHERauxiliary */
 /* ===================================================================== */
 int f2c_dladiv_(bla_double *a, bla_double *b, bla_double *c__, bla_double *d__, bla_double *p, bla_double *q) {
 /* System generated locals */
 bla_double d__1, d__2;
 /* Local variables */
 bla_double s, aa, ab, bb, cc, cd, dd, be, un, ov, eps;
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
 d__1 = bla_d_abs(*a), d__2 = bla_d_abs(*b);
 ab = bla_a_max(d__1,d__2);
 /* Computing MAX */
 d__1 = bla_d_abs(*c__), d__2 = bla_d_abs(*d__);
 cd = bla_a_max(d__1,d__2);
 s = 1.;
 ov = bla_dlamch_("Overflow threshold", (ftnlen)18);
 un = bla_dlamch_("Safe minimum", (ftnlen)12);
 eps = bla_dlamch_("Epsilon", (ftnlen)7);
 be = 2. / (eps * eps);
 if (ab >= ov * .5) {
 aa *= .5;
 bb *= .5;
 s *= 2.;
 }
 if (cd >= ov * .5) {
 cc *= .5;
 dd *= .5;
 s *= .5;
 }
 if (ab <= un * 2. / eps) {
 aa *= be;
 bb *= be;
 s /= be;
 }
 if (cd <= un * 2. / eps) {
 cc *= be;
 dd *= be;
 s *= be;
 }
 if (bla_d_abs(*d__) <= bla_d_abs(*c__)) {
 f2c_dladiv1_(&aa, &bb, &cc, &dd, p, q);
 }
 else {
 f2c_dladiv1_(&bb, &aa, &dd, &cc, p, q);
 *q = -(*q);
 }
 *p *= s;
 *q *= s;
 return 0;
 /* End of DLADIV */
 }
 /* f2c_dladiv_ */
 /* > \ingroup doubleOTHERauxiliary */
 int f2c_dladiv1_(bla_double *a, bla_double *b, bla_double *c__, bla_double *d__, bla_double *p, bla_double *q) {
 bla_double r__, t;
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
 t = 1. / (*c__ + *d__ * r__);
 *p = f2c_dladiv2_(a, b, c__, d__, &r__, &t);
 *a = -(*a);
 *q = f2c_dladiv2_(b, a, c__, d__, &r__, &t);
 return 0;
 /* End of DLADIV1 */
 }
 /* f2c_dladiv1_ */
 /* > \ingroup doubleOTHERauxiliary */
 bla_double f2c_dladiv2_(bla_double *a, bla_double *b, bla_double *c__, bla_double *d__, bla_double *r__, bla_double *t) {
 /* System generated locals */
 bla_double ret_val;
 /* Local variables */
 bla_double br;
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
 if (*r__ != 0.) {
 br = *b * *r__;
 if (br != 0.) {
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
 /* End of DLADIV12 */
 }
 /* f2c_dladiv2_ */
 
