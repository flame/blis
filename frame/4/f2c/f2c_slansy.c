/* f2c_slansy.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_integer c__1 = 1;
 /* > \brief \b SLANSY returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the ele ment of largest absolute value of a bla_real symmetric matrix. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLANSY + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slansy. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slansy. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slansy. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* REAL FUNCTION SLANSY( NORM, UPLO, N, A, LDA, WORK ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER NORM, UPLO */
 /* INTEGER LDA, N */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL A( LDA, * ), WORK( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLANSY returns the value of the one norm, or the Frobenius norm, or */
 /* > the infinity norm, or the element of largest absolute value of a */
 /* > bla_real symmetric matrix A. */
 /* > \endverbatim */
 /* > */
 /* > \return SLANSY */
 /* > \verbatim */
 /* > */
 /* > SLANSY = ( bla_a_max(bla_r_abs(A(i,j))), NORM = 'M' or 'm' */
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
 /* > Specifies the value to be returned in SLANSY as described */
 /* > above. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] UPLO */
 /* > \verbatim */
 /* > UPLO is CHARACTER*1 */
 /* > Specifies whether the upper or lower triangular part of the */
 /* > symmetric matrix A is to be referenced. */
 /* > = 'U': Upper triangular part of A is referenced */
 /* > = 'L': Lower triangular part of A is referenced */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the matrix A. N >= 0. When N = 0, SLANSY is */
 /* > set to zero. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] A */
 /* > \verbatim */
 /* > A is REAL array, dimension (LDA,N) */
 /* > The symmetric matrix A. If UPLO = 'U', the leading n by n */
 /* > upper triangular part of A contains the upper triangular part */
 /* > of the matrix A, and the strictly lower triangular part of A */
 /* > is not referenced. If UPLO = 'L', the leading n by n lower */
 /* > triangular part of A contains the lower triangular part of */
 /* > the matrix A, and the strictly upper triangular part of A is */
 /* > not referenced. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDA */
 /* > \verbatim */
 /* > LDA is INTEGER */
 /* > The leading dimension of the array A. LDA >= bla_a_max(N,1). */
 /* > \endverbatim */
 /* > */
 /* > \param[out] WORK */
 /* > \verbatim */
 /* > WORK is REAL array, dimension (MAX(1,LWORK)), */
 /* > where LWORK >= N when NORM = 'I' or '1' or 'O';
 otherwise, */
 /* > WORK is not referenced. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup realSYauxiliary */
 /* ===================================================================== */
 bla_real f2c_slansy_(char *norm, char *uplo, bla_integer *n, bla_real *a, bla_integer *lda, bla_real * work, ftnlen norm_len, ftnlen uplo_len) {
 /* System generated locals */
 bla_integer a_dim1, a_offset, i__1, i__2;
 bla_real ret_val, r__1;
 /* Builtin functions */
 /* Local variables */
 bla_integer i__, j;
 bla_real sum, ssq[2], absa;
 bla_real value;
 bla_real colssq[2];
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
 /* .. Local Arrays .. */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. External Subroutines .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Parameter adjustments */
 a_dim1 = *lda;
 a_offset = 1 + a_dim1;
 a -= a_offset;
 --work;
 /* Function Body */
 if (*n == 0) {
 value = 0.f;
 }
 else if (bla_lsame_(norm, "M", (ftnlen)1, (ftnlen)1)) {
 /* Find bla_a_max(bla_r_abs(A(i,j))). */
 value = 0.f;
 if (bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
 i__1 = *n;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = j;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 sum = (r__1 = a[i__ + j * a_dim1], bla_r_abs(r__1));
 if (value < sum || f2c_sisnan_(&sum)) {
 value = sum;
 }
 /* L10: */
 }
 /* L20: */
 }
 }
 else {
 i__1 = *n;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *n;
 for (i__ = j;
 i__ <= i__2;
 ++i__) {
 sum = (r__1 = a[i__ + j * a_dim1], bla_r_abs(r__1));
 if (value < sum || f2c_sisnan_(&sum)) {
 value = sum;
 }
 /* L30: */
 }
 /* L40: */
 }
 }
 }
 else if (bla_lsame_(norm, "I", (ftnlen)1, (ftnlen)1) || bla_lsame_(norm, "O", (ftnlen)1, (ftnlen)1) || *(unsigned char *)norm == '1') {
 /* Find normI(A) ( = norm1(A), since A is symmetric). */
 value = 0.f;
 if (bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
 i__1 = *n;
 for (j = 1;
 j <= i__1;
 ++j) {
 sum = 0.f;
 i__2 = j - 1;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 absa = (r__1 = a[i__ + j * a_dim1], bla_r_abs(r__1));
 sum += absa;
 work[i__] += absa;
 /* L50: */
 }
 work[j] = sum + (r__1 = a[j + j * a_dim1], bla_r_abs(r__1));
 /* L60: */
 }
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 sum = work[i__];
 if (value < sum || f2c_sisnan_(&sum)) {
 value = sum;
 }
 /* L70: */
 }
 }
 else {
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 work[i__] = 0.f;
 /* L80: */
 }
 i__1 = *n;
 for (j = 1;
 j <= i__1;
 ++j) {
 sum = work[j] + (r__1 = a[j + j * a_dim1], bla_r_abs(r__1));
 i__2 = *n;
 for (i__ = j + 1;
 i__ <= i__2;
 ++i__) {
 absa = (r__1 = a[i__ + j * a_dim1], bla_r_abs(r__1));
 sum += absa;
 work[i__] += absa;
 /* L90: */
 }
 if (value < sum || f2c_sisnan_(&sum)) {
 value = sum;
 }
 /* L100: */
 }
 }
 }
 else if (bla_lsame_(norm, "F", (ftnlen)1, (ftnlen)1) || bla_lsame_(norm, "E", (ftnlen)1, (ftnlen)1)) {
 /* Find normF(A). */
 /* SSQ(1) is scale */
 /* SSQ(2) is sum-of-squares */
 /* For better accuracy, sum each column separately. */
 ssq[0] = 0.f;
 ssq[1] = 1.f;
 /* Sum off-diagonals */
 if (bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
 i__1 = *n;
 for (j = 2;
 j <= i__1;
 ++j) {
 colssq[0] = 0.f;
 colssq[1] = 1.f;
 i__2 = j - 1;
 f2c_slassq_(&i__2, &a[j * a_dim1 + 1], &c__1, colssq, &colssq[1]);
 f2c_scombssq_(ssq, colssq);
 /* L110: */
 }
 }
 else {
 i__1 = *n - 1;
 for (j = 1;
 j <= i__1;
 ++j) {
 colssq[0] = 0.f;
 colssq[1] = 1.f;
 i__2 = *n - j;
 f2c_slassq_(&i__2, &a[j + 1 + j * a_dim1], &c__1, colssq, &colssq[ 1]);
 f2c_scombssq_(ssq, colssq);
 /* L120: */
 }
 }
 ssq[1] *= 2;
 /* Sum diagonal */
 colssq[0] = 0.f;
 colssq[1] = 1.f;
 i__1 = *lda + 1;
 f2c_slassq_(n, &a[a_offset], &i__1, colssq, &colssq[1]);
 f2c_scombssq_(ssq, colssq);
 value = ssq[0] * sqrt(ssq[1]);
 }
 ret_val = value;
 return ret_val;
 /* End of SLANSY */
 }
 /* f2c_slansy_ */
 
