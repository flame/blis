/* f2c_zungtr.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_integer c__1 = 1;
 static bla_integer c_n1 = -1;
 /* > \brief \b ZUNGTR */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download ZUNGTR + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zungtr. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zungtr. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zungtr. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE ZUNGTR( UPLO, N, A, LDA, TAU, WORK, LWORK, INFO ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER UPLO */
 /* INTEGER INFO, LDA, LWORK, N */
 /* .. */
 /* .. Array Arguments .. */
 /* COMPLEX*16 A( LDA, * ), TAU( * ), WORK( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > ZUNGTR generates a bla_scomplex unitary matrix Q which is defined as the */
 /* > product of n-1 elementary reflectors of order N, as returned by */
 /* > ZHETRD: */
 /* > */
 /* > if UPLO = 'U', Q = H(n-1) . . . H(2) H(1), */
 /* > */
 /* > if UPLO = 'L', Q = H(1) H(2) . . . H(n-1). */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] UPLO */
 /* > \verbatim */
 /* > UPLO is CHARACTER*1 */
 /* > = 'U': Upper triangle of A contains elementary reflectors */
 /* > from ZHETRD;
 */
 /* > = 'L': Lower triangle of A contains elementary reflectors */
 /* > from ZHETRD. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the matrix Q. N >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] A */
 /* > \verbatim */
 /* > A is COMPLEX*16 array, dimension (LDA,N) */
 /* > On entry, the vectors which define the elementary reflectors, */
 /* > as returned by ZHETRD. */
 /* > On exit, the N-by-N unitary matrix Q. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDA */
 /* > \verbatim */
 /* > LDA is INTEGER */
 /* > The leading dimension of the array A. LDA >= N. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] TAU */
 /* > \verbatim */
 /* > TAU is COMPLEX*16 array, dimension (N-1) */
 /* > TAU(i) must contain the scalar factor of the elementary */
 /* > reflector H(i), as returned by ZHETRD. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] WORK */
 /* > \verbatim */
 /* > WORK is COMPLEX*16 array, dimension (MAX(1,LWORK)) */
 /* > On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LWORK */
 /* > \verbatim */
 /* > LWORK is INTEGER */
 /* > The dimension of the array WORK. LWORK >= N-1. */
 /* > For optimum performance LWORK >= (N-1)*NB, where NB is */
 /* > the optimal blocksize. */
 /* > */
 /* > If LWORK = -1, then a workspace query is assumed;
 the routine */
 /* > only calculates the optimal size of the WORK array, returns */
 /* > this value as the first entry of the WORK array, and no error */
 /* > message related to LWORK is issued by XERBLA. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] INFO */
 /* > \verbatim */
 /* > INFO is INTEGER */
 /* > = 0: successful exit */
 /* > < 0: if INFO = -i, the i-th argument had an illegal value */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplex16OTHERcomputational */
 /* ===================================================================== */
 int f2c_zungtr_(char *uplo, bla_integer *n, bla_dcomplex *a, bla_integer *lda, bla_dcomplex *tau, bla_dcomplex *work, bla_integer *lwork, bla_integer *info, ftnlen uplo_len) {
 /* System generated locals */
 bla_integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
 /* Local variables */
 bla_integer i__, j, nb;
 bla_integer iinfo;
 bla_logical upper;
 bla_integer lwkopt;
 bla_logical lquery;
 /* -- LAPACK computational routine -- */
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
 /* Test the input arguments */
 /* Parameter adjustments */
 a_dim1 = *lda;
 a_offset = 1 + a_dim1;
 a -= a_offset;
 --tau;
 --work;
 /* Function Body */
 *info = 0;
 lquery = *lwork == -1;
 upper = bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1);
 if (! upper && ! bla_lsame_(uplo, "L", (ftnlen)1, (ftnlen)1)) {
 *info = -1;
 }
 else if (*n < 0) {
 *info = -2;
 }
 else if (*lda < bla_a_max(1,*n)) {
 *info = -4;
 }
 else /* if(complicated condition) */
 {
 /* Computing MAX */
 i__1 = 1, i__2 = *n - 1;
 if (*lwork < bla_a_max(i__1,i__2) && ! lquery) {
 *info = -7;
 }
 }
 if (*info == 0) {
 if (upper) {
 i__1 = *n - 1;
 i__2 = *n - 1;
 i__3 = *n - 1;
 nb = f2c_ilaenv_(&c__1, "ZUNGQL", " ", &i__1, &i__2, &i__3, &c_n1, (ftnlen)6, (ftnlen)1);
 }
 else {
 i__1 = *n - 1;
 i__2 = *n - 1;
 i__3 = *n - 1;
 nb = f2c_ilaenv_(&c__1, "ZUNGQR", " ", &i__1, &i__2, &i__3, &c_n1, (ftnlen)6, (ftnlen)1);
 }
 /* Computing MAX */
 i__1 = 1, i__2 = *n - 1;
 lwkopt = bla_a_max(i__1,i__2) * nb;
 work[1].real = (bla_double) lwkopt, work[1].imag = 0.;
 }
 if (*info != 0) {
 i__1 = -(*info);
 xerbla_("ZUNGTR", &i__1, (ftnlen)6);
 return 0;
 }
 else if (lquery) {
 return 0;
 }
 /* Quick return if possible */
 if (*n == 0) {
 work[1].real = 1., work[1].imag = 0.;
 return 0;
 }
 if (upper) {
 /* Q was determined by a call to ZHETRD with UPLO = 'U' */
 /* Shift the vectors which define the elementary reflectors one */
 /* column to the left, and set the last row and column of Q to */
 /* those of the unit matrix */
 i__1 = *n - 1;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = j - 1;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = i__ + j * a_dim1;
 i__4 = i__ + (j + 1) * a_dim1;
 a[i__3].real = a[i__4].real, a[i__3].imag = a[i__4].imag;
 /* L10: */
 }
 i__2 = *n + j * a_dim1;
 a[i__2].real = 0., a[i__2].imag = 0.;
 /* L20: */
 }
 i__1 = *n - 1;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 i__2 = i__ + *n * a_dim1;
 a[i__2].real = 0., a[i__2].imag = 0.;
 /* L30: */
 }
 i__1 = *n + *n * a_dim1;
 a[i__1].real = 1., a[i__1].imag = 0.;
 /* Generate Q(1:n-1,1:n-1) */
 i__1 = *n - 1;
 i__2 = *n - 1;
 i__3 = *n - 1;
 f2c_zungql_(&i__1, &i__2, &i__3, &a[a_offset], lda, &tau[1], &work[1], lwork, &iinfo);
 }
 else {
 /* Q was determined by a call to ZHETRD with UPLO = 'L'. */
 /* Shift the vectors which define the elementary reflectors one */
 /* column to the right, and set the first row and column of Q to */
 /* those of the unit matrix */
 for (j = *n;
 j >= 2;
 --j) {
 i__1 = j * a_dim1 + 1;
 a[i__1].real = 0., a[i__1].imag = 0.;
 i__1 = *n;
 for (i__ = j + 1;
 i__ <= i__1;
 ++i__) {
 i__2 = i__ + j * a_dim1;
 i__3 = i__ + (j - 1) * a_dim1;
 a[i__2].real = a[i__3].real, a[i__2].imag = a[i__3].imag;
 /* L40: */
 }
 /* L50: */
 }
 i__1 = a_dim1 + 1;
 a[i__1].real = 1., a[i__1].imag = 0.;
 i__1 = *n;
 for (i__ = 2;
 i__ <= i__1;
 ++i__) {
 i__2 = i__ + a_dim1;
 a[i__2].real = 0., a[i__2].imag = 0.;
 /* L60: */
 }
 if (*n > 1) {
 /* Generate Q(2:n,2:n) */
 i__1 = *n - 1;
 i__2 = *n - 1;
 i__3 = *n - 1;
 f2c_zungqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[1], &work[1], lwork, &iinfo);
 }
 }
 work[1].real = (bla_double) lwkopt, work[1].imag = 0.;
 return 0;
 /* End of ZUNGTR */
 }
 /* f2c_zungtr_ */
 
