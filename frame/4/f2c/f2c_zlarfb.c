/* f2c_zlarfb.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_dcomplex c_b1 = {
1.,0.}
;
 static bla_integer c__1 = 1;
 /* > \brief \b ZLARFB applies a block reflector or its conjugate-transpose to a general rectangular matrix. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download ZLARFB + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlarfb. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlarfb. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlarfb. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE ZLARFB( SIDE, TRANS, DIRECT, STOREV, M, N, K, V, LDV, */
 /* T, LDT, C, LDC, WORK, LDWORK ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER DIRECT, SIDE, STOREV, TRANS */
 /* INTEGER K, LDC, LDT, LDV, LDWORK, M, N */
 /* .. */
 /* .. Array Arguments .. */
 /* COMPLEX*16 C( LDC, * ), T( LDT, * ), V( LDV, * ), */
 /* $ WORK( LDWORK, * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > ZLARFB applies a bla_scomplex block reflector H or its transpose H**H to a */
 /* > bla_scomplex M-by-N matrix C, from either the left or the right. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] SIDE */
 /* > \verbatim */
 /* > SIDE is CHARACTER*1 */
 /* > = 'L': apply H or H**H from the Left */
 /* > = 'R': apply H or H**H from the Right */
 /* > \endverbatim */
 /* > */
 /* > \param[in] TRANS */
 /* > \verbatim */
 /* > TRANS is CHARACTER*1 */
 /* > = 'N': apply H (No transpose) */
 /* > = 'C': apply H**H (Conjugate transpose) */
 /* > \endverbatim */
 /* > */
 /* > \param[in] DIRECT */
 /* > \verbatim */
 /* > DIRECT is CHARACTER*1 */
 /* > Indicates how H is formed from a product of elementary */
 /* > reflectors */
 /* > = 'F': H = H(1) H(2) . . . H(k) (Forward) */
 /* > = 'B': H = H(k) . . . H(2) H(1) (Backward) */
 /* > \endverbatim */
 /* > */
 /* > \param[in] STOREV */
 /* > \verbatim */
 /* > STOREV is CHARACTER*1 */
 /* > Indicates how the vectors which define the elementary */
 /* > reflectors are stored: */
 /* > = 'C': Columnwise */
 /* > = 'R': Rowwise */
 /* > \endverbatim */
 /* > */
 /* > \param[in] M */
 /* > \verbatim */
 /* > M is INTEGER */
 /* > The number of rows of the matrix C. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The number of columns of the matrix C. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] K */
 /* > \verbatim */
 /* > K is INTEGER */
 /* > The order of the matrix T (= the number of elementary */
 /* > reflectors whose product defines the block reflector). */
 /* > If SIDE = 'L', M >= K >= 0;
 */
 /* > if SIDE = 'R', N >= K >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] V */
 /* > \verbatim */
 /* > V is COMPLEX*16 array, dimension */
 /* > (LDV,K) if STOREV = 'C' */
 /* > (LDV,M) if STOREV = 'R' and SIDE = 'L' */
 /* > (LDV,N) if STOREV = 'R' and SIDE = 'R' */
 /* > See Further Details. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDV */
 /* > \verbatim */
 /* > LDV is INTEGER */
 /* > The leading dimension of the array V. */
 /* > If STOREV = 'C' and SIDE = 'L', LDV >= bla_a_max(1,M);
 */
 /* > if STOREV = 'C' and SIDE = 'R', LDV >= bla_a_max(1,N);
 */
 /* > if STOREV = 'R', LDV >= K. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] T */
 /* > \verbatim */
 /* > T is COMPLEX*16 array, dimension (LDT,K) */
 /* > The triangular K-by-K matrix T in the representation of the */
 /* > block reflector. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDT */
 /* > \verbatim */
 /* > LDT is INTEGER */
 /* > The leading dimension of the array T. LDT >= K. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] C */
 /* > \verbatim */
 /* > C is COMPLEX*16 array, dimension (LDC,N) */
 /* > On entry, the M-by-N matrix C. */
 /* > On exit, C is overwritten by H*C or H**H*C or C*H or C*H**H. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDC */
 /* > \verbatim */
 /* > LDC is INTEGER */
 /* > The leading dimension of the array C. LDC >= bla_a_max(1,M). */
 /* > \endverbatim */
 /* > */
 /* > \param[out] WORK */
 /* > \verbatim */
 /* > WORK is COMPLEX*16 array, dimension (LDWORK,K) */
 /* > \endverbatim */
 /* > */
 /* > \param[in] LDWORK */
 /* > \verbatim */
 /* > LDWORK is INTEGER */
 /* > The leading dimension of the array WORK. */
 /* > If SIDE = 'L', LDWORK >= bla_a_max(1,N);
 */
 /* > if SIDE = 'R', LDWORK >= bla_a_max(1,M). */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplex16OTHERauxiliary */
 /* > \par Further Details: */
 /* ===================== */
 /* > */
 /* > \verbatim */
 /* > */
 /* > The shape of the matrix V and the storage of the vectors which define */
 /* > the H(i) is best illustrated by the following example with n = 5 and */
 /* > k = 3. The elements equal to 1 are not stored;
 the corresponding */
 /* > array elements are modified but restored on exit. The rest of the */
 /* > array is not used. */
 /* > */
 /* > DIRECT = 'F' and STOREV = 'C': DIRECT = 'F' and STOREV = 'R': */
 /* > */
 /* > V = ( 1 ) V = ( 1 v1 v1 v1 v1 ) */
 /* > ( v1 1 ) ( 1 v2 v2 v2 ) */
 /* > ( v1 v2 1 ) ( 1 v3 v3 ) */
 /* > ( v1 v2 v3 ) */
 /* > ( v1 v2 v3 ) */
 /* > */
 /* > DIRECT = 'B' and STOREV = 'C': DIRECT = 'B' and STOREV = 'R': */
 /* > */
 /* > V = ( v1 v2 v3 ) V = ( v1 v1 1 ) */
 /* > ( v1 v2 v3 ) ( v2 v2 v2 1 ) */
 /* > ( 1 v2 v3 ) ( v3 v3 v3 v3 1 ) */
 /* > ( 1 v3 ) */
 /* > ( 1 ) */
 /* > \endverbatim */
 /* > */
 /* ===================================================================== */
 int f2c_zlarfb_(char *side, char *trans, char *direct, char * storev, bla_integer *m, bla_integer *n, bla_integer *k, bla_dcomplex *v, bla_integer *ldv, bla_dcomplex *t, bla_integer *ldt, bla_dcomplex *c__, bla_integer * ldc, bla_dcomplex *work, bla_integer *ldwork, ftnlen side_len, ftnlen trans_len, ftnlen direct_len, ftnlen storev_len) {
 /* System generated locals */
 bla_integer c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1, work_offset, i__1, i__2, i__3, i__4, i__5;
 bla_dcomplex z__1, z__2;
 /* Builtin functions */
 /* Local variables */
 bla_integer i__, j;
 char transt[1];
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
 /* Quick return if possible */
 /* Parameter adjustments */
 v_dim1 = *ldv;
 v_offset = 1 + v_dim1;
 v -= v_offset;
 t_dim1 = *ldt;
 t_offset = 1 + t_dim1;
 t -= t_offset;
 c_dim1 = *ldc;
 c_offset = 1 + c_dim1;
 c__ -= c_offset;
 work_dim1 = *ldwork;
 work_offset = 1 + work_dim1;
 work -= work_offset;
 /* Function Body */
 if (*m <= 0 || *n <= 0) {
 return 0;
 }
 if (bla_lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {
 *(unsigned char *)transt = 'C';
 }
 else {
 *(unsigned char *)transt = 'N';
 }
 if (bla_lsame_(storev, "C", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 /* Let V = ( V1 ) (first K rows) */
 /* ( V2 ) */
 /* where V1 is unit lower triangular. */
 if (bla_lsame_(side, "L", (ftnlen)1, (ftnlen)1)) {
 /* Form H * C or H**H * C where C = ( C1 ) */
 /* ( C2 ) */
 /* W := C**H * V = (C1**H * V1 + C2**H * V2) (stored in WORK) */
 /* W := C1**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1], &c__1);
 f2c_zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
 /* L10: */
 }
 /* W := W * V1 */
 ztrmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 if (*m > *k) {
 /* W := W + C2**H * V2 */
 i__1 = *m - *k;
 zgemm_("Conjugate transpose", "No transpose", n, k, &i__1, &c_b1, &c__[*k + 1 + c_dim1], ldc, &v[*k + 1 + v_dim1], ldv, &c_b1, &work[work_offset], ldwork);
 }
 /* W := W * T**H or W * T */
 ztrmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - V * W**H */
 if (*m > *k) {
 /* C2 := C2 - V2 * W**H */
 i__1 = *m - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("No transpose", "Conjugate transpose", &i__1, n, k, &z__1, &v[*k + 1 + v_dim1], ldv, &work[ work_offset], ldwork, &c_b1, &c__[*k + 1 + c_dim1] , ldc);
 }
 /* W := W * V1**H */
 ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", n, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 /* C1 := C1 - W**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = j + i__ * c_dim1;
 i__4 = j + i__ * c_dim1;
 bla_d_cnjg(&z__2, &work[i__ + j * work_dim1]);
 z__1.real = c__[i__4].real - z__2.real, z__1.imag = c__[i__4].imag - z__2.imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L20: */
 }
 /* L30: */
 }
 }
 else if (bla_lsame_(side, "R", (ftnlen)1, (ftnlen)1)) {
 /* Form C * H or C * H**H where C = ( C1 C2 ) */
 /* W := C * V = (C1*V1 + C2*V2) (stored in WORK) */
 /* W := C1 */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j * work_dim1 + 1], &c__1);
 /* L40: */
 }
 /* W := W * V1 */
 ztrmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 if (*n > *k) {
 /* W := W + C2 * V2 */
 i__1 = *n - *k;
 zgemm_("No transpose", "No transpose", m, k, &i__1, &c_b1, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k + 1 + v_dim1], ldv, &c_b1, &work[work_offset], ldwork);
 }
 /* W := W * T or W * T**H */
 ztrmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - W * V**H */
 if (*n > *k) {
 /* C2 := C2 - W * V2**H */
 i__1 = *n - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("No transpose", "Conjugate transpose", m, &i__1, k, &z__1, &work[work_offset], ldwork, &v[*k + 1 + v_dim1], ldv, &c_b1, &c__[(*k + 1) * c_dim1 + 1], ldc);
 }
 /* W := W * V1**H */
 ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", m, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 /* C1 := C1 - W */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = i__ + j * c_dim1;
 i__4 = i__ + j * c_dim1;
 i__5 = i__ + j * work_dim1;
 z__1.real = c__[i__4].real - work[i__5].real, z__1.imag = c__[ i__4].imag - work[i__5].imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L50: */
 }
 /* L60: */
 }
 }
 }
 else {
 /* Let V = ( V1 ) */
 /* ( V2 ) (last K rows) */
 /* where V2 is unit upper triangular. */
 if (bla_lsame_(side, "L", (ftnlen)1, (ftnlen)1)) {
 /* Form H * C or H**H * C where C = ( C1 ) */
 /* ( C2 ) */
 /* W := C**H * V = (C1**H * V1 + C2**H * V2) (stored in WORK) */
 /* W := C2**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j * work_dim1 + 1], &c__1);
 f2c_zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
 /* L70: */
 }
 /* W := W * V2 */
 ztrmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b1, &v[*m - *k + 1 + v_dim1], ldv, &work[work_offset], ldwork);
 if (*m > *k) {
 /* W := W + C1**H * V1 */
 i__1 = *m - *k;
 zgemm_("Conjugate transpose", "No transpose", n, k, &i__1, &c_b1, &c__[c_offset], ldc, &v[v_offset], ldv, & c_b1, &work[work_offset], ldwork);
 }
 /* W := W * T**H or W * T */
 ztrmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - V * W**H */
 if (*m > *k) {
 /* C1 := C1 - V1 * W**H */
 i__1 = *m - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("No transpose", "Conjugate transpose", &i__1, n, k, &z__1, &v[v_offset], ldv, &work[work_offset], ldwork, &c_b1, &c__[c_offset], ldc);
 }
 /* W := W * V2**H */
 ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", n, k, &c_b1, &v[*m - *k + 1 + v_dim1], ldv, &work[ work_offset], ldwork);
 /* C2 := C2 - W**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = *m - *k + j + i__ * c_dim1;
 i__4 = *m - *k + j + i__ * c_dim1;
 bla_d_cnjg(&z__2, &work[i__ + j * work_dim1]);
 z__1.real = c__[i__4].real - z__2.real, z__1.imag = c__[i__4].imag - z__2.imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L80: */
 }
 /* L90: */
 }
 }
 else if (bla_lsame_(side, "R", (ftnlen)1, (ftnlen)1)) {
 /* Form C * H or C * H**H where C = ( C1 C2 ) */
 /* W := C * V = (C1*V1 + C2*V2) (stored in WORK) */
 /* W := C2 */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[ j * work_dim1 + 1], &c__1);
 /* L100: */
 }
 /* W := W * V2 */
 ztrmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b1, &v[*n - *k + 1 + v_dim1], ldv, &work[work_offset], ldwork);
 if (*n > *k) {
 /* W := W + C1 * V1 */
 i__1 = *n - *k;
 zgemm_("No transpose", "No transpose", m, k, &i__1, &c_b1, &c__[c_offset], ldc, &v[v_offset], ldv, &c_b1, & work[work_offset], ldwork) ;
 }
 /* W := W * T or W * T**H */
 ztrmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - W * V**H */
 if (*n > *k) {
 /* C1 := C1 - W * V1**H */
 i__1 = *n - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("No transpose", "Conjugate transpose", m, &i__1, k, &z__1, &work[work_offset], ldwork, &v[v_offset], ldv, &c_b1, &c__[c_offset], ldc);
 }
 /* W := W * V2**H */
 ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", m, k, &c_b1, &v[*n - *k + 1 + v_dim1], ldv, &work[ work_offset], ldwork);
 /* C2 := C2 - W */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = i__ + (*n - *k + j) * c_dim1;
 i__4 = i__ + (*n - *k + j) * c_dim1;
 i__5 = i__ + j * work_dim1;
 z__1.real = c__[i__4].real - work[i__5].real, z__1.imag = c__[ i__4].imag - work[i__5].imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L110: */
 }
 /* L120: */
 }
 }
 }
 }
 else if (bla_lsame_(storev, "R", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 /* Let V = ( V1 V2 ) (V1: first K columns) */
 /* where V1 is unit upper triangular. */
 if (bla_lsame_(side, "L", (ftnlen)1, (ftnlen)1)) {
 /* Form H * C or H**H * C where C = ( C1 ) */
 /* ( C2 ) */
 /* W := C**H * V**H = (C1**H * V1**H + C2**H * V2**H) (stored in WORK) */
 /* W := C1**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1], &c__1);
 f2c_zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
 /* L130: */
 }
 /* W := W * V1**H */
 ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", n, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 if (*m > *k) {
 /* W := W + C2**H * V2**H */
 i__1 = *m - *k;
 zgemm_("Conjugate transpose", "Conjugate transpose", n, k, &i__1, &c_b1, &c__[*k + 1 + c_dim1], ldc, &v[(*k + 1) * v_dim1 + 1], ldv, &c_b1, &work[work_offset] , ldwork);
 }
 /* W := W * T**H or W * T */
 ztrmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - V**H * W**H */
 if (*m > *k) {
 /* C2 := C2 - V2**H * W**H */
 i__1 = *m - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("Conjugate transpose", "Conjugate transpose", & i__1, n, k, &z__1, &v[(*k + 1) * v_dim1 + 1], ldv, &work[work_offset], ldwork, &c_b1, &c__[*k + 1 + c_dim1], ldc);
 }
 /* W := W * V1 */
 ztrmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 /* C1 := C1 - W**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = j + i__ * c_dim1;
 i__4 = j + i__ * c_dim1;
 bla_d_cnjg(&z__2, &work[i__ + j * work_dim1]);
 z__1.real = c__[i__4].real - z__2.real, z__1.imag = c__[i__4].imag - z__2.imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L140: */
 }
 /* L150: */
 }
 }
 else if (bla_lsame_(side, "R", (ftnlen)1, (ftnlen)1)) {
 /* Form C * H or C * H**H where C = ( C1 C2 ) */
 /* W := C * V**H = (C1*V1**H + C2*V2**H) (stored in WORK) */
 /* W := C1 */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j * work_dim1 + 1], &c__1);
 /* L160: */
 }
 /* W := W * V1**H */
 ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", m, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 if (*n > *k) {
 /* W := W + C2 * V2**H */
 i__1 = *n - *k;
 zgemm_("No transpose", "Conjugate transpose", m, k, &i__1, &c_b1, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[(*k + 1) * v_dim1 + 1], ldv, &c_b1, &work[work_offset] , ldwork);
 }
 /* W := W * T or W * T**H */
 ztrmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - W * V */
 if (*n > *k) {
 /* C2 := C2 - W * V2 */
 i__1 = *n - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("No transpose", "No transpose", m, &i__1, k, &z__1, &work[work_offset], ldwork, &v[(*k + 1) * v_dim1 + 1], ldv, &c_b1, &c__[(*k + 1) * c_dim1 + 1], ldc);
 }
 /* W := W * V1 */
 ztrmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
 /* C1 := C1 - W */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = i__ + j * c_dim1;
 i__4 = i__ + j * c_dim1;
 i__5 = i__ + j * work_dim1;
 z__1.real = c__[i__4].real - work[i__5].real, z__1.imag = c__[ i__4].imag - work[i__5].imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L170: */
 }
 /* L180: */
 }
 }
 }
 else {
 /* Let V = ( V1 V2 ) (V2: last K columns) */
 /* where V2 is unit lower triangular. */
 if (bla_lsame_(side, "L", (ftnlen)1, (ftnlen)1)) {
 /* Form H * C or H**H * C where C = ( C1 ) */
 /* ( C2 ) */
 /* W := C**H * V**H = (C1**H * V1**H + C2**H * V2**H) (stored in WORK) */
 /* W := C2**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j * work_dim1 + 1], &c__1);
 f2c_zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
 /* L190: */
 }
 /* W := W * V2**H */
 ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", n, k, &c_b1, &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[ work_offset], ldwork);
 if (*m > *k) {
 /* W := W + C1**H * V1**H */
 i__1 = *m - *k;
 zgemm_("Conjugate transpose", "Conjugate transpose", n, k, &i__1, &c_b1, &c__[c_offset], ldc, &v[v_offset], ldv, &c_b1, &work[work_offset], ldwork);
 }
 /* W := W * T**H or W * T */
 ztrmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - V**H * W**H */
 if (*m > *k) {
 /* C1 := C1 - V1**H * W**H */
 i__1 = *m - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("Conjugate transpose", "Conjugate transpose", & i__1, n, k, &z__1, &v[v_offset], ldv, &work[ work_offset], ldwork, &c_b1, &c__[c_offset], ldc);
 }
 /* W := W * V2 */
 ztrmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b1, &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[ work_offset], ldwork);
 /* C2 := C2 - W**H */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = *m - *k + j + i__ * c_dim1;
 i__4 = *m - *k + j + i__ * c_dim1;
 bla_d_cnjg(&z__2, &work[i__ + j * work_dim1]);
 z__1.real = c__[i__4].real - z__2.real, z__1.imag = c__[i__4].imag - z__2.imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L200: */
 }
 /* L210: */
 }
 }
 else if (bla_lsame_(side, "R", (ftnlen)1, (ftnlen)1)) {
 /* Form C * H or C * H**H where C = ( C1 C2 ) */
 /* W := C * V**H = (C1*V1**H + C2*V2**H) (stored in WORK) */
 /* W := C2 */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 zcopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[ j * work_dim1 + 1], &c__1);
 /* L220: */
 }
 /* W := W * V2**H */
 ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", m, k, &c_b1, &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[ work_offset], ldwork);
 if (*n > *k) {
 /* W := W + C1 * V1**H */
 i__1 = *n - *k;
 zgemm_("No transpose", "Conjugate transpose", m, k, &i__1, &c_b1, &c__[c_offset], ldc, &v[v_offset], ldv, & c_b1, &work[work_offset], ldwork);
 }
 /* W := W * T or W * T**H */
 ztrmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b1, &t[ t_offset], ldt, &work[work_offset], ldwork);
 /* C := C - W * V */
 if (*n > *k) {
 /* C1 := C1 - W * V1 */
 i__1 = *n - *k;
 z__1.real = -1., z__1.imag = -0.;
 zgemm_("No transpose", "No transpose", m, &i__1, k, &z__1, &work[work_offset], ldwork, &v[v_offset], ldv, & c_b1, &c__[c_offset], ldc) ;
 }
 /* W := W * V2 */
 ztrmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b1, &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[ work_offset], ldwork);
 /* C1 := C1 - W */
 i__1 = *k;
 for (j = 1;
 j <= i__1;
 ++j) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 i__3 = i__ + (*n - *k + j) * c_dim1;
 i__4 = i__ + (*n - *k + j) * c_dim1;
 i__5 = i__ + j * work_dim1;
 z__1.real = c__[i__4].real - work[i__5].real, z__1.imag = c__[ i__4].imag - work[i__5].imag;
 c__[i__3].real = z__1.real, c__[i__3].imag = z__1.imag;
 /* L230: */
 }
 /* L240: */
 }
 }
 }
 }
 return 0;
 /* End of ZLARFB */
 }
 /* f2c_zlarfb_ */
 
