/* f2c_slanst.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_integer c__1 = 1;
 /* > \brief \b SLANST returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the ele ment of largest absolute value of a bla_real symmetric tridiagonal matrix. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLANST + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slanst. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slanst. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slanst. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* REAL FUNCTION SLANST( NORM, N, D, E ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER NORM */
 /* INTEGER N */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL D( * ), E( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLANST returns the value of the one norm, or the Frobenius norm, or */
 /* > the infinity norm, or the element of largest absolute value of a */
 /* > bla_real symmetric tridiagonal matrix A. */
 /* > \endverbatim */
 /* > */
 /* > \return SLANST */
 /* > \verbatim */
 /* > */
 /* > SLANST = ( bla_a_max(bla_r_abs(A(i,j))), NORM = 'M' or 'm' */
 /* > ( */
 /* > ( norm1(A), NORM = '1', 'O' or 'o' */
 /* > ( */
 /* > ( normI(A), NORM = 'I' or 'i' */
 /* > ( */
 /* > ( normF(A), NORM = 'F', 'f', 'E' or 'e' */
 /* > */
 /* > where norm1 denotes the one norm of a matrix (maximum column sum), */
 /* > normI denotes the infinity norm of a matrix (maximum row sum) and */
 /* > normF denotes the Frobenius norm of a matrix (square root of sum of */
 /* > squares). Note that bla_a_max(bla_r_abs(A(i,j))) is not a consistent matrix norm. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] NORM */
 /* > \verbatim */
 /* > NORM is CHARACTER*1 */
 /* > Specifies the value to be returned in SLANST as described */
 /* > above. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the matrix A. N >= 0. When N = 0, SLANST is */
 /* > set to zero. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] D */
 /* > \verbatim */
 /* > D is REAL array, dimension (N) */
 /* > The diagonal elements of A. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] E */
 /* > \verbatim */
 /* > E is REAL array, dimension (N-1) */
 /* > The (n-1) sub-diagonal or super-diagonal elements of A. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* ===================================================================== */
 bla_real f2c_slanst_(char *norm, bla_integer *n, bla_real *d__, bla_real *e, ftnlen norm_len) {
 /* System generated locals */
 bla_integer i__1;
 bla_real ret_val, r__1, r__2, r__3;
 /* Builtin functions */
 /* Local variables */
 bla_integer i__;
 bla_real sum, scale;
 bla_real anorm;
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
 /* .. External Subroutines .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Parameter adjustments */
 --e;
 --d__;
 /* Function Body */
 if (*n <= 0) {
 anorm = 0.f;
 }
 else if (bla_lsame_(norm, "M", (ftnlen)1, (ftnlen)1)) {
 /* Find bla_a_max(bla_r_abs(A(i,j))). */
 anorm = (r__1 = d__[*n], bla_r_abs(r__1));
 i__1 = *n - 1;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 sum = (r__1 = d__[i__], bla_r_abs(r__1));
 if (anorm < sum || f2c_sisnan_(&sum)) {
 anorm = sum;
 }
 sum = (r__1 = e[i__], bla_r_abs(r__1));
 if (anorm < sum || f2c_sisnan_(&sum)) {
 anorm = sum;
 }
 /* L10: */
 }
 }
 else if (bla_lsame_(norm, "O", (ftnlen)1, (ftnlen)1) || *(unsigned char *) norm == '1' || bla_lsame_(norm, "I", (ftnlen)1, (ftnlen)1)) {
 /* Find norm1(A). */
 if (*n == 1) {
 anorm = bla_r_abs(d__[1]);
 }
 else {
 anorm = bla_r_abs(d__[1]) + bla_r_abs(e[1]);
 sum = (r__1 = e[*n - 1], bla_r_abs(r__1)) + (r__2 = d__[*n], bla_r_abs(r__2));
 if (anorm < sum || f2c_sisnan_(&sum)) {
 anorm = sum;
 }
 i__1 = *n - 1;
 for (i__ = 2;
 i__ <= i__1;
 ++i__) {
 sum = (r__1 = d__[i__], bla_r_abs(r__1)) + (r__2 = e[i__], bla_r_abs(r__2) ) + (r__3 = e[i__ - 1], bla_r_abs(r__3));
 if (anorm < sum || f2c_sisnan_(&sum)) {
 anorm = sum;
 }
 /* L20: */
 }
 }
 }
 else if (bla_lsame_(norm, "F", (ftnlen)1, (ftnlen)1) || bla_lsame_(norm, "E", (ftnlen)1, (ftnlen)1)) {
 /* Find normF(A). */
 scale = 0.f;
 sum = 1.f;
 if (*n > 1) {
 i__1 = *n - 1;
 f2c_slassq_(&i__1, &e[1], &c__1, &scale, &sum);
 sum *= 2;
 }
 f2c_slassq_(n, &d__[1], &c__1, &scale, &sum);
 anorm = scale * sqrt(sum);
 }
 ret_val = anorm;
 return ret_val;
 /* End of SLANST */
 }
 /* f2c_slanst_ */
 
