/* f2c_slaev2.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLAEV2 computes the eigenvalues and eigenvectors of a 2-by-2 symmetric/Hermitian matrix. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLAEV2 + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slaev2. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slaev2. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slaev2. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SLAEV2( A, B, C, RT1, RT2, CS1, SN1 ) */
 /* .. Scalar Arguments .. */
 /* REAL A, B, C, CS1, RT1, RT2, SN1 */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix */
 /* > [ A B ] */
 /* > [ B C ]. */
 /* > On return, RT1 is the eigenvalue of larger absolute value, RT2 is the */
 /* > eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right */
 /* > eigenvector for RT1, giving the decomposition */
 /* > */
 /* > [ CS1 SN1 ] [ A B ] [ CS1 -SN1 ] = [ RT1 0 ] */
 /* > [-SN1 CS1 ] [ B C ] [ SN1 CS1 ] [ 0 RT2 ]. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] A */
 /* > \verbatim */
 /* > A is REAL */
 /* > The (1,1) element of the 2-by-2 matrix. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] B */
 /* > \verbatim */
 /* > B is REAL */
 /* > The (1,2) element and the conjugate of the (2,1) element of */
 /* > the 2-by-2 matrix. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] C */
 /* > \verbatim */
 /* > C is REAL */
 /* > The (2,2) element of the 2-by-2 matrix. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] RT1 */
 /* > \verbatim */
 /* > RT1 is REAL */
 /* > The eigenvalue of larger absolute value. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] RT2 */
 /* > \verbatim */
 /* > RT2 is REAL */
 /* > The eigenvalue of smaller absolute value. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] CS1 */
 /* > \verbatim */
 /* > CS1 is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[out] SN1 */
 /* > \verbatim */
 /* > SN1 is REAL */
 /* > The vector (CS1, SN1) is a unit right eigenvector for RT1. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* > \par Further Details: */
 /* ===================== */
 /* > */
 /* > \verbatim */
 /* > */
 /* > RT1 is accurate to a few ulps barring over/underflow. */
 /* > */
 /* > RT2 may be inaccurate if there is massive cancellation in the */
 /* > determinant A*C-B*B;
 higher precision or correctly rounded or */
 /* > correctly truncated arithmetic would be needed to compute RT2 */
 /* > accurately in all cases. */
 /* > */
 /* > CS1 and SN1 are accurate to a few ulps barring over/underflow. */
 /* > */
 /* > Overflow is possible only if RT1 is within a factor of 5 of overflow. */
 /* > Underflow is harmless if the input data is 0 or exceeds */
 /* > underflow_threshold / macheps. */
 /* > \endverbatim */
 /* > */
 /* ===================================================================== */
 int f2c_slaev2_(bla_real *a, bla_real *b, bla_real *c__, bla_real *rt1, bla_real * rt2, bla_real *cs1, bla_real *sn1) {
 /* System generated locals */
 bla_real r__1;
 /* Builtin functions */
 /* Local variables */
 bla_real ab, df, cs, ct, tb, sm, tn, rt, adf, acs;
 bla_integer sgn1, sgn2;
 bla_real acmn, acmx;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Parameters .. */
 /* .. */
 /* .. Local Scalars .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Compute the eigenvalues */
 sm = *a + *c__;
 df = *a - *c__;
 adf = bla_r_abs(df);
 tb = *b + *b;
 ab = bla_r_abs(tb);
 if (bla_r_abs(*a) > bla_r_abs(*c__)) {
 acmx = *a;
 acmn = *c__;
 }
 else {
 acmx = *c__;
 acmn = *a;
 }
 if (adf > ab) {
 /* Computing 2nd power */
 r__1 = ab / adf;
 rt = adf * sqrt(r__1 * r__1 + 1.f);
 }
 else if (adf < ab) {
 /* Computing 2nd power */
 r__1 = adf / ab;
 rt = ab * sqrt(r__1 * r__1 + 1.f);
 }
 else {
 /* Includes case AB=ADF=0 */
 rt = ab * sqrt(2.f);
 }
 if (sm < 0.f) {
 *rt1 = (sm - rt) * .5f;
 sgn1 = -1;
 /* Order of execution important. */
 /* To get fully accurate smaller eigenvalue, */
 /* next line needs to be executed in higher precision. */
 *rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
 }
 else if (sm > 0.f) {
 *rt1 = (sm + rt) * .5f;
 sgn1 = 1;
 /* Order of execution important. */
 /* To get fully accurate smaller eigenvalue, */
 /* next line needs to be executed in higher precision. */
 *rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
 }
 else {
 /* Includes case RT1 = RT2 = 0 */
 *rt1 = rt * .5f;
 *rt2 = rt * -.5f;
 sgn1 = 1;
 }
 /* Compute the eigenvector */
 if (df >= 0.f) {
 cs = df + rt;
 sgn2 = 1;
 }
 else {
 cs = df - rt;
 sgn2 = -1;
 }
 acs = bla_r_abs(cs);
 if (acs > ab) {
 ct = -tb / cs;
 *sn1 = 1.f / sqrt(ct * ct + 1.f);
 *cs1 = ct * *sn1;
 }
 else {
 if (ab == 0.f) {
 *cs1 = 1.f;
 *sn1 = 0.f;
 }
 else {
 tn = -cs / tb;
 *cs1 = 1.f / sqrt(tn * tn + 1.f);
 *sn1 = tn * *cs1;
 }
 }
 if (sgn1 == sgn2) {
 tn = *cs1;
 *cs1 = -(*sn1);
 *sn1 = tn;
 }
 return 0;
 /* End of SLAEV2 */
 }
 /* f2c_slaev2_ */
 
