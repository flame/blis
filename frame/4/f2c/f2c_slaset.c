/* f2c_slaset.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLASET initializes the off-diagonal elements and the diagonal elements of a matrix to given val ues. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLASET + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slaset. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slaset. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slaset. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SLASET( UPLO, M, N, ALPHA, BETA, A, LDA ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER UPLO */
 /* INTEGER LDA, M, N */
 /* REAL ALPHA, BETA */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL A( LDA, * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SLASET initializes an m-by-n matrix A to BETA on the diagonal and */
 /* > ALPHA on the offdiagonals. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] UPLO */
 /* > \verbatim */
 /* > UPLO is CHARACTER*1 */
 /* > Specifies the part of the matrix A to be set. */
 /* > = 'U': Upper triangular part is set;
 the strictly lower */
 /* > triangular part of A is not changed. */
 /* > = 'L': Lower triangular part is set;
 the strictly upper */
 /* > triangular part of A is not changed. */
 /* > Otherwise: All of the matrix A is set. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] M */
 /* > \verbatim */
 /* > M is INTEGER */
 /* > The number of rows of the matrix A. M >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The number of columns of the matrix A. N >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] ALPHA */
 /* > \verbatim */
 /* > ALPHA is REAL */
 /* > The constant to which the offdiagonal elements are to be set. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] BETA */
 /* > \verbatim */
 /* > BETA is REAL */
 /* > The constant to which the diagonal elements are to be set. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] A */
 /* > \verbatim */
 /* > A is REAL array, dimension (LDA,N) */
 /* > On exit, the leading m-by-n submatrix of A is set as follows: */
 /* > */
 /* > if UPLO = 'U', A(i,j) = ALPHA, 1<=i<=j-1, 1<=j<=n, */
 /* > if UPLO = 'L', A(i,j) = ALPHA, j+1<=i<=m, 1<=j<=n, */
 /* > otherwise, A(i,j) = ALPHA, 1<=i<=m, 1<=j<=n, i.ne.j, */
 /* > */
 /* > and, for all UPLO, A(i,i) = BETA, 1<=i<=bla_a_min(m,n). */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDA */
 /* > \verbatim */
 /* > LDA is INTEGER */
 /* > The leading dimension of the array A. LDA >= bla_a_max(1,M). */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* ===================================================================== */
 int f2c_slaset_(char *uplo, bla_integer *m, bla_integer *n, bla_real *alpha, bla_real *beta, bla_real *a, bla_integer *lda, ftnlen uplo_len) {
 /* System generated locals */
 bla_integer a_dim1, a_offset, i__1, i__2, i__3;
 /* Local variables */
 bla_integer i__, j;
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
 /* .. External Functions .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Parameter adjustments */
 a_dim1 = *lda;
 a_offset = 1 + a_dim1;
 a -= a_offset;
 /* Function Body */
 if (bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
 /* Set the strictly upper triangular or trapezoidal part of the */
 /* array to ALPHA. */
 i__1 = *n;
 for (j = 2;
 j <= i__1;
 ++j) {
 /* Computing MIN */
 i__3 = j - 1;
 i__2 = bla_a_min(i__3,*m);
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 a[i__ + j * a_dim1] = *alpha;
 /* L10: */
 }
 /* L20: */
 }
 }
 else if (bla_lsame_(uplo, "L", (ftnlen)1, (ftnlen)1)) {
 /* Set the strictly lower triangular or trapezoidal part of the */
 /* array to ALPHA. */
 i__1 = bla_a_min(*m,*n);
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *m;
 for (i__ = j + 1;
 i__ <= i__2;
 ++i__) {
 a[i__ + j * a_dim1] = *alpha;
 /* L30: */
 }
 /* L40: */
 }
 }
 else {
 /* Set the leading m-by-n submatrix to ALPHA. */
 i__1 = *n;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 a[i__ + j * a_dim1] = *alpha;
 /* L50: */
 }
 /* L60: */
 }
 }
 /* Set the first bla_a_min(M,N) diagonal elements to BETA. */
 i__1 = bla_a_min(*m,*n);
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 a[i__ + i__ * a_dim1] = *beta;
 /* L70: */
 }
 return 0;
 /* End of SLASET */
 }
 /* f2c_slaset_ */
 
