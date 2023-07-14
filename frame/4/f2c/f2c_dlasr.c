/* f2c_dlasr.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b DLASR applies a sequence of plane rotations to a general rectangular matrix. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download DLASR + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasr.f "> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasr.f "> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasr.f "> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE DLASR( SIDE, PIVOT, DIRECT, M, N, C, S, A, LDA ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER DIRECT, PIVOT, SIDE */
 /* INTEGER LDA, M, N */
 /* .. */
 /* .. Array Arguments .. */
 /* DOUBLE PRECISION A( LDA, * ), C( * ), S( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > DLASR applies a sequence of plane rotations to a bla_real matrix A, */
 /* > from either the left or the right. */
 /* > */
 /* > When SIDE = 'L', the transformation takes the form */
 /* > */
 /* > A := P*A */
 /* > */
 /* > and when SIDE = 'R', the transformation takes the form */
 /* > */
 /* > A := A*P**T */
 /* > */
 /* > where P is an orthogonal matrix consisting of a sequence of z plane */
 /* > rotations, with z = M when SIDE = 'L' and z = N when SIDE = 'R', */
 /* > and P**T is the transpose of P. */
 /* > */
 /* > When DIRECT = 'F' (Forward sequence), then */
 /* > */
 /* > P = P(z-1) * ... * P(2) * P(1) */
 /* > */
 /* > and when DIRECT = 'B' (Backward sequence), then */
 /* > */
 /* > P = P(1) * P(2) * ... * P(z-1) */
 /* > */
 /* > where P(k) is a plane rotation matrix defined by the 2-by-2 rotation */
 /* > */
 /* > R(k) = ( c(k) s(k) ) */
 /* > = ( -s(k) c(k) ). */
 /* > */
 /* > When PIVOT = 'V' (Variable pivot), the rotation is performed */
 /* > for the plane (k,k+1), i.e., P(k) has the form */
 /* > */
 /* > P(k) = ( 1 ) */
 /* > ( ... ) */
 /* > ( 1 ) */
 /* > ( c(k) s(k) ) */
 /* > ( -s(k) c(k) ) */
 /* > ( 1 ) */
 /* > ( ... ) */
 /* > ( 1 ) */
 /* > */
 /* > where R(k) appears as a rank-2 modification to the identity matrix in */
 /* > rows and columns k and k+1. */
 /* > */
 /* > When PIVOT = 'T' (Top pivot), the rotation is performed for the */
 /* > plane (1,k+1), so P(k) has the form */
 /* > */
 /* > P(k) = ( c(k) s(k) ) */
 /* > ( 1 ) */
 /* > ( ... ) */
 /* > ( 1 ) */
 /* > ( -s(k) c(k) ) */
 /* > ( 1 ) */
 /* > ( ... ) */
 /* > ( 1 ) */
 /* > */
 /* > where R(k) appears in rows and columns 1 and k+1. */
 /* > */
 /* > Similarly, when PIVOT = 'B' (Bottom pivot), the rotation is */
 /* > performed for the plane (k,z), giving P(k) the form */
 /* > */
 /* > P(k) = ( 1 ) */
 /* > ( ... ) */
 /* > ( 1 ) */
 /* > ( c(k) s(k) ) */
 /* > ( 1 ) */
 /* > ( ... ) */
 /* > ( 1 ) */
 /* > ( -s(k) c(k) ) */
 /* > */
 /* > where R(k) appears in rows and columns k and z. The rotations are */
 /* > performed without ever forming P(k) explicitly. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] SIDE */
 /* > \verbatim */
 /* > SIDE is CHARACTER*1 */
 /* > Specifies whether the plane rotation matrix P is applied to */
 /* > A on the left or the right. */
 /* > = 'L': Left, compute A := P*A */
 /* > = 'R': Right, compute A:= A*P**T */
 /* > \endverbatim */
 /* > */
 /* > \param[in] PIVOT */
 /* > \verbatim */
 /* > PIVOT is CHARACTER*1 */
 /* > Specifies the plane for which P(k) is a plane rotation */
 /* > matrix. */
 /* > = 'V': Variable pivot, the plane (k,k+1) */
 /* > = 'T': Top pivot, the plane (1,k+1) */
 /* > = 'B': Bottom pivot, the plane (k,z) */
 /* > \endverbatim */
 /* > */
 /* > \param[in] DIRECT */
 /* > \verbatim */
 /* > DIRECT is CHARACTER*1 */
 /* > Specifies whether P is a forward or backward sequence of */
 /* > plane rotations. */
 /* > = 'F': Forward, P = P(z-1)*...*P(2)*P(1) */
 /* > = 'B': Backward, P = P(1)*P(2)*...*P(z-1) */
 /* > \endverbatim */
 /* > */
 /* > \param[in] M */
 /* > \verbatim */
 /* > M is INTEGER */
 /* > The number of rows of the matrix A. If m <= 1, an immediate */
 /* > return is effected. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The number of columns of the matrix A. If n <= 1, an */
 /* > immediate return is effected. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] C */
 /* > \verbatim */
 /* > C is DOUBLE PRECISION array, dimension */
 /* > (M-1) if SIDE = 'L' */
 /* > (N-1) if SIDE = 'R' */
 /* > The cosines c(k) of the plane rotations. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] S */
 /* > \verbatim */
 /* > S is DOUBLE PRECISION array, dimension */
 /* > (M-1) if SIDE = 'L' */
 /* > (N-1) if SIDE = 'R' */
 /* > The sines s(k) of the plane rotations. The 2-by-2 plane */
 /* > rotation part of the matrix P(k), R(k), has the form */
 /* > R(k) = ( c(k) s(k) ) */
 /* > ( -s(k) c(k) ). */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] A */
 /* > \verbatim */
 /* > A is DOUBLE PRECISION array, dimension (LDA,N) */
 /* > The M-by-N matrix A. On exit, A is overwritten by P*A if */
 /* > SIDE = 'L' or by A*P**T if SIDE = 'R'. */
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
 int f2c_dlasr_(char *side, char *pivot, char *direct, bla_integer *m, bla_integer *n, bla_double *c__, bla_double *s, bla_double *a, bla_integer * lda, ftnlen side_len, ftnlen pivot_len, ftnlen direct_len) {
 /* System generated locals */
 bla_integer a_dim1, a_offset, i__1, i__2;
 /* Local variables */
 bla_integer i__, j, info;
 bla_double temp;
 bla_double ctemp, stemp;
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
 /* Test the input parameters */
 /* Parameter adjustments */
 --c__;
 --s;
 a_dim1 = *lda;
 a_offset = 1 + a_dim1;
 a -= a_offset;
 /* Function Body */
 info = 0;
 if (! (bla_lsame_(side, "L", (ftnlen)1, (ftnlen)1) || bla_lsame_(side, "R", (ftnlen)1, (ftnlen)1))) {
 info = 1;
 }
 else if (! (bla_lsame_(pivot, "V", (ftnlen)1, (ftnlen)1) || bla_lsame_(pivot, "T", (ftnlen)1, (ftnlen)1) || bla_lsame_(pivot, "B", (ftnlen)1, (ftnlen)1))) {
 info = 2;
 }
 else if (! (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1) || bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1))) {
 info = 3;
 }
 else if (*m < 0) {
 info = 4;
 }
 else if (*n < 0) {
 info = 5;
 }
 else if (*lda < bla_a_max(1,*m)) {
 info = 9;
 }
 if (info != 0) {
 xerbla_("DLASR ", &info, (ftnlen)6);
 return 0;
 }
 /* Quick return if possible */
 if (*m == 0 || *n == 0) {
 return 0;
 }
 if (bla_lsame_(side, "L", (ftnlen)1, (ftnlen)1)) {
 /* Form P * A */
 if (bla_lsame_(pivot, "V", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 i__1 = *m - 1;
 for (j = 1;
 j <= i__1;
 ++j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 temp = a[j + 1 + i__ * a_dim1];
 a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp * a[j + i__ * a_dim1];
 a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j + i__ * a_dim1];
 /* L10: */
 }
 }
 /* L20: */
 }
 }
 else if (bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
 for (j = *m - 1;
 j >= 1;
 --j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 temp = a[j + 1 + i__ * a_dim1];
 a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp * a[j + i__ * a_dim1];
 a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j + i__ * a_dim1];
 /* L30: */
 }
 }
 /* L40: */
 }
 }
 }
 else if (bla_lsame_(pivot, "T", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 i__1 = *m;
 for (j = 2;
 j <= i__1;
 ++j) {
 ctemp = c__[j - 1];
 stemp = s[j - 1];
 if (ctemp != 1. || stemp != 0.) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 temp = a[j + i__ * a_dim1];
 a[j + i__ * a_dim1] = ctemp * temp - stemp * a[ i__ * a_dim1 + 1];
 a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[ i__ * a_dim1 + 1];
 /* L50: */
 }
 }
 /* L60: */
 }
 }
 else if (bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
 for (j = *m;
 j >= 2;
 --j) {
 ctemp = c__[j - 1];
 stemp = s[j - 1];
 if (ctemp != 1. || stemp != 0.) {
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 temp = a[j + i__ * a_dim1];
 a[j + i__ * a_dim1] = ctemp * temp - stemp * a[ i__ * a_dim1 + 1];
 a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[ i__ * a_dim1 + 1];
 /* L70: */
 }
 }
 /* L80: */
 }
 }
 }
 else if (bla_lsame_(pivot, "B", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 i__1 = *m - 1;
 for (j = 1;
 j <= i__1;
 ++j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__2 = *n;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 temp = a[j + i__ * a_dim1];
 a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1] + ctemp * temp;
 a[*m + i__ * a_dim1] = ctemp * a[*m + i__ * a_dim1] - stemp * temp;
 /* L90: */
 }
 }
 /* L100: */
 }
 }
 else if (bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
 for (j = *m - 1;
 j >= 1;
 --j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__1 = *n;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 temp = a[j + i__ * a_dim1];
 a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1] + ctemp * temp;
 a[*m + i__ * a_dim1] = ctemp * a[*m + i__ * a_dim1] - stemp * temp;
 /* L110: */
 }
 }
 /* L120: */
 }
 }
 }
 }
 else if (bla_lsame_(side, "R", (ftnlen)1, (ftnlen)1)) {
 /* Form A * P**T */
 if (bla_lsame_(pivot, "V", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 i__1 = *n - 1;
 for (j = 1;
 j <= i__1;
 ++j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 temp = a[i__ + (j + 1) * a_dim1];
 a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp * a[i__ + j * a_dim1];
 a[i__ + j * a_dim1] = stemp * temp + ctemp * a[ i__ + j * a_dim1];
 /* L130: */
 }
 }
 /* L140: */
 }
 }
 else if (bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
 for (j = *n - 1;
 j >= 1;
 --j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__1 = *m;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 temp = a[i__ + (j + 1) * a_dim1];
 a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp * a[i__ + j * a_dim1];
 a[i__ + j * a_dim1] = stemp * temp + ctemp * a[ i__ + j * a_dim1];
 /* L150: */
 }
 }
 /* L160: */
 }
 }
 }
 else if (bla_lsame_(pivot, "T", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 i__1 = *n;
 for (j = 2;
 j <= i__1;
 ++j) {
 ctemp = c__[j - 1];
 stemp = s[j - 1];
 if (ctemp != 1. || stemp != 0.) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 temp = a[i__ + j * a_dim1];
 a[i__ + j * a_dim1] = ctemp * temp - stemp * a[ i__ + a_dim1];
 a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ + a_dim1];
 /* L170: */
 }
 }
 /* L180: */
 }
 }
 else if (bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
 for (j = *n;
 j >= 2;
 --j) {
 ctemp = c__[j - 1];
 stemp = s[j - 1];
 if (ctemp != 1. || stemp != 0.) {
 i__1 = *m;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 temp = a[i__ + j * a_dim1];
 a[i__ + j * a_dim1] = ctemp * temp - stemp * a[ i__ + a_dim1];
 a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ + a_dim1];
 /* L190: */
 }
 }
 /* L200: */
 }
 }
 }
 else if (bla_lsame_(pivot, "B", (ftnlen)1, (ftnlen)1)) {
 if (bla_lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
 i__1 = *n - 1;
 for (j = 1;
 j <= i__1;
 ++j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__2 = *m;
 for (i__ = 1;
 i__ <= i__2;
 ++i__) {
 temp = a[i__ + j * a_dim1];
 a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1] + ctemp * temp;
 a[i__ + *n * a_dim1] = ctemp * a[i__ + *n * a_dim1] - stemp * temp;
 /* L210: */
 }
 }
 /* L220: */
 }
 }
 else if (bla_lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
 for (j = *n - 1;
 j >= 1;
 --j) {
 ctemp = c__[j];
 stemp = s[j];
 if (ctemp != 1. || stemp != 0.) {
 i__1 = *m;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 temp = a[i__ + j * a_dim1];
 a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1] + ctemp * temp;
 a[i__ + *n * a_dim1] = ctemp * a[i__ + *n * a_dim1] - stemp * temp;
 /* L230: */
 }
 }
 /* L240: */
 }
 }
 }
 }
 return 0;
 /* End of DLASR */
 }
 /* f2c_dlasr_ */
 
