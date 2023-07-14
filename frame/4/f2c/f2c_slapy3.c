/* f2c_slapy3.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLAPY3 returns sqrt(x2+y2+z2). */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLAPY3 + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slapy3. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slapy3. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slapy3. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* REAL FUNCTION SLAPY3( X, Y, Z ) */
 /* .. Scalar Arguments .. */
 /* REAL X, Y, Z */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause */
 /* > unnecessary overflow. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] X */
 /* > \verbatim */
 /* > X is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[in] Y */
 /* > \verbatim */
 /* > Y is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[in] Z */
 /* > \verbatim */
 /* > Z is REAL */
 /* > X, Y and Z specify the values x, y and z. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* ===================================================================== */
 bla_real f2c_slapy3_(bla_real *x, bla_real *y, bla_real *z__) {
 /* System generated locals */
 bla_real ret_val, r__1, r__2, r__3;
 /* Builtin functions */
 /* Local variables */
 bla_real w, xabs, yabs, zabs;
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
 xabs = bla_r_abs(*x);
 yabs = bla_r_abs(*y);
 zabs = bla_r_abs(*z__);
 /* Computing MAX */
 r__1 = bla_a_max(xabs,yabs);
 w = bla_a_max(r__1,zabs);
 if (w == 0.f) {
 /* W can be zero for bla_a_max(0,nan,0) */
 /* adding all three entries together will make sure */
 /* NaN will not disappear. */
 ret_val = xabs + yabs + zabs;
 }
 else {
 /* Computing 2nd power */
 r__1 = xabs / w;
 /* Computing 2nd power */
 r__2 = yabs / w;
 /* Computing 2nd power */
 r__3 = zabs / w;
 ret_val = w * sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
 }
 return ret_val;
 /* End of SLAPY3 */
 }
 /* f2c_slapy3_ */
 
