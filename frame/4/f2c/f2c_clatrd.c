/* f2c_clatrd.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_scomplex c_b1 = {
0.f,0.f}
;
 static bla_scomplex c_b2 = {
1.f,0.f}
;
 static bla_integer c__1 = 1;
 /* > \brief \b CLATRD reduces the first nb rows and columns of a symmetric/Hermitian matrix A to bla_real tridiago nal form by an unitary similarity transformation. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download CLATRD + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/clatrd. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/clatrd. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/clatrd. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE CLATRD( UPLO, N, NB, A, LDA, E, TAU, W, LDW ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER UPLO */
 /* INTEGER LDA, LDW, N, NB */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL E( * ) */
 /* COMPLEX A( LDA, * ), TAU( * ), W( LDW, * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > CLATRD reduces NB rows and columns of a bla_scomplex Hermitian matrix A to */
 /* > Hermitian tridiagonal form by a unitary similarity */
 /* > transformation Q**H * A * Q, and returns the matrices V and W which are */
 /* > needed to apply the transformation to the unreduced part of A. */
 /* > */
 /* > If UPLO = 'U', CLATRD reduces the last NB rows and columns of a */
 /* > matrix, of which the upper triangle is supplied;
 */
 /* > if UPLO = 'L', CLATRD reduces the first NB rows and columns of a */
 /* > matrix, of which the lower triangle is supplied. */
 /* > */
 /* > This is an auxiliary routine called by CHETRD. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] UPLO */
 /* > \verbatim */
 /* > UPLO is CHARACTER*1 */
 /* > Specifies whether the upper or lower triangular part of the */
 /* > Hermitian matrix A is stored: */
 /* > = 'U': Upper triangular */
 /* > = 'L': Lower triangular */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the matrix A. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] NB */
 /* > \verbatim */
 /* > NB is INTEGER */
 /* > The number of rows and columns to be reduced. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] A */
 /* > \verbatim */
 /* > A is COMPLEX array, dimension (LDA,N) */
 /* > On entry, the Hermitian matrix A. If UPLO = 'U', the leading */
 /* > n-by-n upper triangular part of A contains the upper */
 /* > triangular part of the matrix A, and the strictly lower */
 /* > triangular part of A is not referenced. If UPLO = 'L', the */
 /* > leading n-by-n lower triangular part of A contains the lower */
 /* > triangular part of the matrix A, and the strictly upper */
 /* > triangular part of A is not referenced. */
 /* > On exit: */
 /* > if UPLO = 'U', the last NB columns have been reduced to */
 /* > tridiagonal form, with the diagonal elements overwriting */
 /* > the diagonal elements of A;
 the elements above the diagonal */
 /* > with the array TAU, represent the unitary matrix Q as a */
 /* > product of elementary reflectors;
 */
 /* > if UPLO = 'L', the first NB columns have been reduced to */
 /* > tridiagonal form, with the diagonal elements overwriting */
 /* > the diagonal elements of A;
 the elements below the diagonal */
 /* > with the array TAU, represent the unitary matrix Q as a */
 /* > product of elementary reflectors. */
 /* > See Further Details. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDA */
 /* > \verbatim */
 /* > LDA is INTEGER */
 /* > The leading dimension of the array A. LDA >= bla_a_max(1,N). */
 /* > \endverbatim */
 /* > */
 /* > \param[out] E */
 /* > \verbatim */
 /* > E is REAL array, dimension (N-1) */
 /* > If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal */
 /* > elements of the last NB columns of the reduced matrix;
 */
 /* > if UPLO = 'L', E(1:nb) contains the subdiagonal elements of */
 /* > the first NB columns of the reduced matrix. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] TAU */
 /* > \verbatim */
 /* > TAU is COMPLEX array, dimension (N-1) */
 /* > The scalar factors of the elementary reflectors, stored in */
 /* > TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'. */
 /* > See Further Details. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] W */
 /* > \verbatim */
 /* > W is COMPLEX array, dimension (LDW,NB) */
 /* > The n-by-nb matrix W required to update the unreduced part */
 /* > of A. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDW */
 /* > \verbatim */
 /* > LDW is INTEGER */
 /* > The leading dimension of the array W. LDW >= bla_a_max(1,N). */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplexOTHERauxiliary */
 /* > \par Further Details: */
 /* ===================== */
 /* > */
 /* > \verbatim */
 /* > */
 /* > If UPLO = 'U', the matrix Q is represented as a product of elementary */
 /* > reflectors */
 /* > */
 /* > Q = H(n) H(n-1) . . . H(n-nb+1). */
 /* > */
 /* > Each H(i) has the form */
 /* > */
 /* > H(i) = I - tau * v * v**H */
 /* > */
 /* > where tau is a bla_scomplex scalar, and v is a bla_scomplex vector with */
 /* > v(i:n) = 0 and v(i-1) = 1;
 v(1:i-1) is stored on exit in A(1:i-1,i), */
 /* > and tau in TAU(i-1). */
 /* > */
 /* > If UPLO = 'L', the matrix Q is represented as a product of elementary */
 /* > reflectors */
 /* > */
 /* > Q = H(1) H(2) . . . H(nb). */
 /* > */
 /* > Each H(i) has the form */
 /* > */
 /* > H(i) = I - tau * v * v**H */
 /* > */
 /* > where tau is a bla_scomplex scalar, and v is a bla_scomplex vector with */
 /* > v(1:i) = 0 and v(i+1) = 1;
 v(i+1:n) is stored on exit in A(i+1:n,i), */
 /* > and tau in TAU(i). */
 /* > */
 /* > The elements of the vectors v together form the n-by-nb matrix V */
 /* > which is needed, with W, to apply the transformation to the unreduced */
 /* > part of the matrix, using a Hermitian rank-2k update of the form: */
 /* > A := A - V*W**H - W*V**H. */
 /* > */
 /* > The contents of A on exit are illustrated by the following examples */
 /* > with n = 5 and nb = 2: */
 /* > */
 /* > if UPLO = 'U': if UPLO = 'L': */
 /* > */
 /* > ( a a a v4 v5 ) ( d ) */
 /* > ( a a v4 v5 ) ( 1 d ) */
 /* > ( a 1 v5 ) ( v1 1 a ) */
 /* > ( d 1 ) ( v1 v2 a a ) */
 /* > ( d ) ( v1 v2 a a a ) */
 /* > */
 /* > where d denotes a diagonal element of the reduced matrix, a denotes */
 /* > an element of the original matrix that is unchanged, and vi denotes */
 /* > an element of the vector defining H(i). */
 /* > \endverbatim */
 /* > */
 /* ===================================================================== */
 int f2c_clatrd_(char *uplo, bla_integer *n, bla_integer *nb, bla_scomplex *a, bla_integer *lda, bla_real *e, bla_scomplex *tau, bla_scomplex *w, bla_integer *ldw, ftnlen uplo_len) {
 /* System generated locals */
 bla_integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;
 bla_real r__1;
 bla_scomplex q__1, q__2, q__3, q__4;
 /* Local variables */
 bla_integer i__, iw;
 bla_scomplex alpha;
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
 /* .. External Subroutines .. */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 /* Quick return if possible */
 /* Parameter adjustments */
 a_dim1 = *lda;
 a_offset = 1 + a_dim1;
 a -= a_offset;
 --e;
 --tau;
 w_dim1 = *ldw;
 w_offset = 1 + w_dim1;
 w -= w_offset;
 /* Function Body */
 if (*n <= 0) {
 return 0;
 }
 if (bla_lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
 /* Reduce last NB columns of upper triangle */
 i__1 = *n - *nb + 1;
 for (i__ = *n;
 i__ >= i__1;
 --i__) {
 iw = i__ - *n + *nb;
 if (i__ < *n) {
 /* Update A(1:i,i) */
 i__2 = i__ + i__ * a_dim1;
 i__3 = i__ + i__ * a_dim1;
 r__1 = a[i__3].real;
 a[i__2].real = r__1, a[i__2].imag = 0.f;
 i__2 = *n - i__;
 f2c_clacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
 i__2 = *n - i__;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__, &i__2, &q__1, &a[(i__ + 1) * a_dim1 + 1], lda, &w[i__ + (iw + 1) * w_dim1], ldw, & c_b2, &a[i__ * a_dim1 + 1], &c__1);
 i__2 = *n - i__;
 f2c_clacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
 i__2 = *n - i__;
 f2c_clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
 i__2 = *n - i__;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__, &i__2, &q__1, &w[(iw + 1) * w_dim1 + 1], ldw, &a[i__ + (i__ + 1) * a_dim1], lda, & c_b2, &a[i__ * a_dim1 + 1], &c__1);
 i__2 = *n - i__;
 f2c_clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
 i__2 = i__ + i__ * a_dim1;
 i__3 = i__ + i__ * a_dim1;
 r__1 = a[i__3].real;
 a[i__2].real = r__1, a[i__2].imag = 0.f;
 }
 if (i__ > 1) {
 /* Generate elementary reflector H(i) to annihilate */
 /* A(1:i-2,i) */
 i__2 = i__ - 1 + i__ * a_dim1;
 alpha.real = a[i__2].real, alpha.imag = a[i__2].imag;
 i__2 = i__ - 1;
 f2c_clarfg_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &tau[i__ - 1]);
 i__2 = i__ - 1;
 e[i__2] = alpha.real;
 i__2 = i__ - 1 + i__ * a_dim1;
 a[i__2].real = 1.f, a[i__2].imag = 0.f;
 /* Compute W(1:i-1,i) */
 i__2 = i__ - 1;
 chemv_("Upper", &i__2, &c_b2, &a[a_offset], lda, &a[i__ * a_dim1 + 1], &c__1, &c_b1, &w[iw * w_dim1 + 1], &c__1);
 if (i__ < *n) {
 i__2 = i__ - 1;
 i__3 = *n - i__;
 cgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &w[(iw + 1) * w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1], & c__1, &c_b1, &w[i__ + 1 + iw * w_dim1], &c__1);
 i__2 = i__ - 1;
 i__3 = *n - i__;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__2, &i__3, &q__1, &a[(i__ + 1) * a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], & c__1, &c_b2, &w[iw * w_dim1 + 1], &c__1);
 i__2 = i__ - 1;
 i__3 = *n - i__;
 cgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[( i__ + 1) * a_dim1 + 1], lda, &a[i__ * a_dim1 + 1], &c__1, &c_b1, &w[i__ + 1 + iw * w_dim1], &c__1);
 i__2 = i__ - 1;
 i__3 = *n - i__;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__2, &i__3, &q__1, &w[(iw + 1) * w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1], & c__1, &c_b2, &w[iw * w_dim1 + 1], &c__1);
 }
 i__2 = i__ - 1;
 cscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
 q__3.real = -.5f, q__3.imag = -0.f;
 i__2 = i__ - 1;
 q__2.real = q__3.real * tau[i__2].real - q__3.imag * tau[i__2].imag, q__2.imag = q__3.real * tau[i__2].imag + q__3.imag * tau[i__2].real;
 i__3 = i__ - 1;
 f2c_cdotc_(&q__4, &i__3, &w[iw * w_dim1 + 1], &c__1, &a[i__ * a_dim1 + 1], &c__1);
 q__1.real = q__2.real * q__4.real - q__2.imag * q__4.imag, q__1.imag = q__2.real * q__4.imag + q__2.imag * q__4.real;
 alpha.real = q__1.real, alpha.imag = q__1.imag;
 i__2 = i__ - 1;
 caxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &w[iw * w_dim1 + 1], &c__1);
 }
 /* L10: */
 }
 }
 else {
 /* Reduce first NB columns of lower triangle */
 i__1 = *nb;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 /* Update A(i:n,i) */
 i__2 = i__ + i__ * a_dim1;
 i__3 = i__ + i__ * a_dim1;
 r__1 = a[i__3].real;
 a[i__2].real = r__1, a[i__2].imag = 0.f;
 i__2 = i__ - 1;
 f2c_clacgv_(&i__2, &w[i__ + w_dim1], ldw);
 i__2 = *n - i__ + 1;
 i__3 = i__ - 1;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + a_dim1], lda, &w[i__ + w_dim1], ldw, &c_b2, &a[i__ + i__ * a_dim1], & c__1);
 i__2 = i__ - 1;
 f2c_clacgv_(&i__2, &w[i__ + w_dim1], ldw);
 i__2 = i__ - 1;
 f2c_clacgv_(&i__2, &a[i__ + a_dim1], lda);
 i__2 = *n - i__ + 1;
 i__3 = i__ - 1;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__2, &i__3, &q__1, &w[i__ + w_dim1], ldw, &a[i__ + a_dim1], lda, &c_b2, &a[i__ + i__ * a_dim1], & c__1);
 i__2 = i__ - 1;
 f2c_clacgv_(&i__2, &a[i__ + a_dim1], lda);
 i__2 = i__ + i__ * a_dim1;
 i__3 = i__ + i__ * a_dim1;
 r__1 = a[i__3].real;
 a[i__2].real = r__1, a[i__2].imag = 0.f;
 if (i__ < *n) {
 /* Generate elementary reflector H(i) to annihilate */
 /* A(i+2:n,i) */
 i__2 = i__ + 1 + i__ * a_dim1;
 alpha.real = a[i__2].real, alpha.imag = a[i__2].imag;
 i__2 = *n - i__;
 /* Computing MIN */
 i__3 = i__ + 2;
 f2c_clarfg_(&i__2, &alpha, &a[bla_a_min(i__3,*n) + i__ * a_dim1], &c__1, &tau[i__]);
 i__2 = i__;
 e[i__2] = alpha.real;
 i__2 = i__ + 1 + i__ * a_dim1;
 a[i__2].real = 1.f, a[i__2].imag = 0.f;
 /* Compute W(i+1:n,i) */
 i__2 = *n - i__;
 chemv_("Lower", &i__2, &c_b2, &a[i__ + 1 + (i__ + 1) * a_dim1] , lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b1, &w[ i__ + 1 + i__ * w_dim1], &c__1);
 i__2 = *n - i__;
 i__3 = i__ - 1;
 cgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &w[i__ + 1 + w_dim1], ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, & c_b1, &w[i__ * w_dim1 + 1], &c__1);
 i__2 = *n - i__;
 i__3 = i__ - 1;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + 1 + a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, &c_b2, &w[ i__ + 1 + i__ * w_dim1], &c__1);
 i__2 = *n - i__;
 i__3 = i__ - 1;
 cgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[i__ + 1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, & c_b1, &w[i__ * w_dim1 + 1], &c__1);
 i__2 = *n - i__;
 i__3 = i__ - 1;
 q__1.real = -1.f, q__1.imag = -0.f;
 cgemv_("No transpose", &i__2, &i__3, &q__1, &w[i__ + 1 + w_dim1], ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b2, &w[ i__ + 1 + i__ * w_dim1], &c__1);
 i__2 = *n - i__;
 cscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
 q__3.real = -.5f, q__3.imag = -0.f;
 i__2 = i__;
 q__2.real = q__3.real * tau[i__2].real - q__3.imag * tau[i__2].imag, q__2.imag = q__3.real * tau[i__2].imag + q__3.imag * tau[i__2].real;
 i__3 = *n - i__;
 f2c_cdotc_(&q__4, &i__3, &w[i__ + 1 + i__ * w_dim1], &c__1, &a[ i__ + 1 + i__ * a_dim1], &c__1);
 q__1.real = q__2.real * q__4.real - q__2.imag * q__4.imag, q__1.imag = q__2.real * q__4.imag + q__2.imag * q__4.real;
 alpha.real = q__1.real, alpha.imag = q__1.imag;
 i__2 = *n - i__;
 caxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &w[ i__ + 1 + i__ * w_dim1], &c__1);
 }
 /* L20: */
 }
 }
 return 0;
 /* End of CLATRD */
 }
 /* f2c_clatrd_ */
 
