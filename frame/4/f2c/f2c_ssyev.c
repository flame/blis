/* f2c_ssyev.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_integer c__1 = 1;
 static bla_integer c_n1 = -1;
 static bla_integer c__0 = 0;
 static bla_real c_b17 = 1.f;
 /* > \brief <b> SSYEV computes the eigenvalues and, optionally, the left and/or right eigenvectors for SY matr ices</b> */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SSYEV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ssyev.f "> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ssyev.f "> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ssyev.f "> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SSYEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER JOBZ, UPLO */
 /* INTEGER INFO, LDA, LWORK, N */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL A( LDA, * ), W( * ), WORK( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SSYEV computes all eigenvalues and, optionally, eigenvectors of a */
 /* > bla_real symmetric matrix A. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] JOBZ */
 /* > \verbatim */
 /* > JOBZ is CHARACTER*1 */
 /* > = 'N': Compute eigenvalues only;
 */
 /* > = 'V': Compute eigenvalues and eigenvectors. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] UPLO */
 /* > \verbatim */
 /* > UPLO is CHARACTER*1 */
 /* > = 'U': Upper triangle of A is stored;
 */
 /* > = 'L': Lower triangle of A is stored. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the matrix A. N >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] A */
 /* > \verbatim */
 /* > A is REAL array, dimension (LDA, N) */
 /* > On entry, the symmetric matrix A. If UPLO = 'U', the */
 /* > leading N-by-N upper triangular part of A contains the */
 /* > upper triangular part of the matrix A. If UPLO = 'L', */
 /* > the leading N-by-N lower triangular part of A contains */
 /* > the lower triangular part of the matrix A. */
 /* > On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
 /* > orthonormal eigenvectors of the matrix A. */
 /* > If JOBZ = 'N', then on exit the lower triangle (if UPLO='L') */
 /* > or the upper triangle (if UPLO='U') of A, including the */
 /* > diagonal, is destroyed. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDA */
 /* > \verbatim */
 /* > LDA is INTEGER */
 /* > The leading dimension of the array A. LDA >= bla_a_max(1,N). */
 /* > \endverbatim */
 /* > */
 /* > \param[out] W */
 /* > \verbatim */
 /* > W is REAL array, dimension (N) */
 /* > If INFO = 0, the eigenvalues in ascending order. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] WORK */
 /* > \verbatim */
 /* > WORK is REAL array, dimension (MAX(1,LWORK)) */
 /* > On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LWORK */
 /* > \verbatim */
 /* > LWORK is INTEGER */
 /* > The length of the array WORK. LWORK >= bla_a_max(1,3*N-1). */
 /* > For optimal efficiency, LWORK >= (NB+2)*N, */
 /* > where NB is the blocksize for SSYTRD returned by ILAENV. */
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
 /* > > 0: if INFO = i, the algorithm failed to converge;
 i */
 /* > off-diagonal elements of an intermediate tridiagonal */
 /* > form did not converge to zero. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup realSYeigen */
 /* ===================================================================== */
 int f2c_ssyev_(char *jobz, char *uplo, bla_integer *n, bla_real *a, bla_integer *lda, bla_real *w, bla_real *work, bla_integer *lwork, bla_real *rwork, bla_integer *info, ftnlen jobz_len, ftnlen uplo_len) {
 /* System generated locals */
 bla_integer a_dim1, a_offset, i__1, i__2;
 bla_real r__1;
 /* Builtin functions */
 /* Local variables */
 bla_integer nb;
 bla_real eps;
 bla_integer inde;
 bla_real anrm;
 bla_integer imax;
 bla_real rmin, rmax, sigma;
 bla_integer iinfo;
 bla_logical lower, wantz;
 bla_integer iscale;
 bla_real safmin;
 bla_real bignum;
 bla_integer indtau, indwrk;
 bla_integer llwork;
 bla_real smlnum;
 bla_integer lwkopt;
 bla_logical lquery;
 /* -- LAPACK driver routine -- */
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
 /* Test the input parameters. */
 /* Parameter adjustments */
 a_dim1 = *lda;
 a_offset = 1 + a_dim1;
 a -= a_offset;
 --w;
 --work;
 /* Function Body */
 wantz = bla_lsame_(jobz, "V", (ftnlen)1, (ftnlen)1);
 lower = bla_lsame_(uplo, "L", (ftnlen)1, (ftnlen)1);
 lquery = *lwork == -1;
 *info = 0;
 if (! (wantz || bla_lsame_(jobz, "N", (ftnlen)1, (ftnlen)1))) {
 *info = -1;
 }
 else if (! (lower || bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1))) {
 *info = -2;
 }
 else if (*n < 0) {
 *info = -3;
 }
 else if (*lda < bla_a_max(1,*n)) {
 *info = -5;
 }
 if (*info == 0) {
 nb = f2c_ilaenv_(&c__1, "SSYTRD", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
 /* Computing MAX */
 i__1 = 1, i__2 = (nb + 2) * *n;
 lwkopt = bla_a_max(i__1,i__2);
 work[1] = (bla_real) lwkopt;
 /* Computing MAX */
 i__1 = 1, i__2 = *n * 3 - 1;
 if (*lwork < bla_a_max(i__1,i__2) && ! lquery) {
 *info = -8;
 }
 }
 if (*info != 0) {
 i__1 = -(*info);
 xerbla_("SSYEV ", &i__1, (ftnlen)6);
 return 0;
 }
 else if (lquery) {
 return 0;
 }
 /* Quick return if possible */
 if (*n == 0) {
 return 0;
 }
 if (*n == 1) {
 w[1] = a[a_dim1 + 1];
 work[1] = 2.f;
 if (wantz) {
 a[a_dim1 + 1] = 1.f;
 }
 return 0;
 }
 /* Get machine constants. */
 safmin = bla_slamch_("Safe minimum", (ftnlen)12);
 eps = bla_slamch_("Precision", (ftnlen)9);
 smlnum = safmin / eps;
 bignum = 1.f / smlnum;
 rmin = sqrt(smlnum);
 rmax = sqrt(bignum);
 /* Scale matrix to allowable range, if necessary. */
 anrm = f2c_slansy_("M", uplo, n, &a[a_offset], lda, &work[1], (ftnlen)1, (ftnlen)1);
 iscale = 0;
 if (anrm > 0.f && anrm < rmin) {
 iscale = 1;
 sigma = rmin / anrm;
 }
 else if (anrm > rmax) {
 iscale = 1;
 sigma = rmax / anrm;
 }
 if (iscale == 1) {
 f2c_slascl_(uplo, &c__0, &c__0, &c_b17, &sigma, n, n, &a[a_offset], lda, info, (ftnlen)1);
 }
 /* Call SSYTRD to reduce symmetric matrix to tridiagonal form. */
 inde = 1;
 indtau = inde + *n;
 indwrk = indtau + *n;
 llwork = *lwork - indwrk + 1;
 f2c_ssytrd_(uplo, n, &a[a_offset], lda, &w[1], &work[inde], &work[indtau], & work[indwrk], &llwork, &iinfo, (ftnlen)1);
 /* For eigenvalues only, call SSTERF. For eigenvectors, first call */
 /* SORGTR to generate the orthogonal matrix, then call SSTEQR. */
 if (! wantz) {
 f2c_ssterf_(n, &w[1], &work[inde], info);
 }
 else {
 f2c_sorgtr_(uplo, n, &a[a_offset], lda, &work[indtau], &work[indwrk], & llwork, &iinfo, (ftnlen)1);
 f2c_ssteqr_(jobz, n, &w[1], &work[inde], &a[a_offset], lda, &work[indtau], info, (ftnlen)1);
 }
 /* If matrix was scaled, then rescale eigenvalues appropriately. */
 if (iscale == 1) {
 if (*info == 0) {
 imax = *n;
 }
 else {
 imax = *info - 1;
 }
 r__1 = 1.f / sigma;
 sscal_(&imax, &r__1, &w[1], &c__1);
 }
 /* Set WORK(1) to optimal workspace size. */
 work[1] = (bla_real) lwkopt;
 return 0;
 /* End of SSYEV */
 }
 /* f2c_ssyev_ */
 
