/* f2c_ssterf.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_integer c__0 = 0;
 static bla_integer c__1 = 1;
 static bla_real c_b32 = 1.f;
 /* > \brief \b SSTERF */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SSTERF + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ssterf. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ssterf. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ssterf. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* SUBROUTINE SSTERF( N, D, E, INFO ) */
 /* .. Scalar Arguments .. */
 /* INTEGER INFO, N */
 /* .. */
 /* .. Array Arguments .. */
 /* REAL D( * ), E( * ) */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > SSTERF computes all eigenvalues of a symmetric tridiagonal matrix */
 /* > using the Pal-Walker-Kahan variant of the QL or QR algorithm. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] N */
 /* > \verbatim */
 /* > N is INTEGER */
 /* > The order of the matrix. N >= 0. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] D */
 /* > \verbatim */
 /* > D is REAL array, dimension (N) */
 /* > On entry, the n diagonal elements of the tridiagonal matrix. */
 /* > On exit, if INFO = 0, the eigenvalues in ascending order. */
 /* > \endverbatim */
 /* > */
 /* > \param[in,out] E */
 /* > \verbatim */
 /* > E is REAL array, dimension (N-1) */
 /* > On entry, the (n-1) subdiagonal elements of the tridiagonal */
 /* > matrix. */
 /* > On exit, E has been destroyed. */
 /* > \endverbatim */
 /* > */
 /* > \param[out] INFO */
 /* > \verbatim */
 /* > INFO is INTEGER */
 /* > = 0: successful exit */
 /* > < 0: if INFO = -i, the i-th argument had an illegal value */
 /* > > 0: the algorithm failed to find all of the eigenvalues in */
 /* > a total of 30*N iterations;
 if INFO = i, then i */
 /* > elements of E have not converged to zero. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup auxOTHERcomputational */
 /* ===================================================================== */
 int f2c_ssterf_(bla_integer *n, bla_real *d__, bla_real *e, bla_integer *info) {
 /* System generated locals */
 bla_integer i__1;
 bla_real r__1, r__2, r__3;
 /* Builtin functions */
 /* Local variables */
 bla_real c__;
 bla_integer i__, l, m;
 bla_real p, r__, s;
 bla_integer l1;
 bla_real bb, rt1, rt2, eps, rte;
 bla_integer lsv;
 bla_real eps2, oldc;
 bla_integer lend, jtot;
 bla_real gamma, alpha, sigma, anorm;
 bla_integer iscale;
 bla_real oldgam;
 bla_real safmin;
 bla_real safmax;
 bla_integer lendsv;
 bla_real ssfmin;
 bla_integer nmaxit;
 bla_real ssfmax;
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
 /* Test the input parameters. */
 /* Parameter adjustments */
 --e;
 --d__;
 /* Function Body */
 *info = 0;
 /* Quick return if possible */
 if (*n < 0) {
 *info = -1;
 i__1 = -(*info);
 xerbla_("SSTERF", &i__1, (ftnlen)6);
 return 0;
 }
 if (*n <= 1) {
 return 0;
 }
 /* Determine the unit roundoff for this environment. */
 eps = bla_slamch_("E", (ftnlen)1);
 /* Computing 2nd power */
 r__1 = eps;
 eps2 = r__1 * r__1;
 safmin = bla_slamch_("S", (ftnlen)1);
 safmax = 1.f / safmin;
 ssfmax = sqrt(safmax) / 3.f;
 ssfmin = sqrt(safmin) / eps2;
 /* Compute the eigenvalues of the tridiagonal matrix. */
 nmaxit = *n * 30;
 sigma = 0.f;
 jtot = 0;
 /* Determine where the matrix splits and choose QL or QR iteration */
 /* for each block, according to whether top or bottom diagonal */
 /* element is smaller. */
 l1 = 1;
 L10: if (l1 > *n) {
 goto L170;
 }
 if (l1 > 1) {
 e[l1 - 1] = 0.f;
 }
 i__1 = *n - 1;
 for (m = l1;
 m <= i__1;
 ++m) {
 if ((r__3 = e[m], bla_r_abs(r__3)) <= sqrt((r__1 = d__[m], bla_r_abs(r__1))) * sqrt((r__2 = d__[m + 1], bla_r_abs(r__2))) * eps) {
 e[m] = 0.f;
 goto L30;
 }
 /* L20: */
 }
 m = *n;
 L30: l = l1;
 lsv = l;
 lend = m;
 lendsv = lend;
 l1 = m + 1;
 if (lend == l) {
 goto L10;
 }
 /* Scale submatrix in rows and columns L to LEND */
 i__1 = lend - l + 1;
 anorm = f2c_slanst_("M", &i__1, &d__[l], &e[l], (ftnlen)1);
 iscale = 0;
 if (anorm == 0.f) {
 goto L10;
 }
 if (anorm > ssfmax) {
 iscale = 1;
 i__1 = lend - l + 1;
 f2c_slascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &d__[l], n, info, (ftnlen)1);
 i__1 = lend - l;
 f2c_slascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &e[l], n, info, (ftnlen)1);
 }
 else if (anorm < ssfmin) {
 iscale = 2;
 i__1 = lend - l + 1;
 f2c_slascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &d__[l], n, info, (ftnlen)1);
 i__1 = lend - l;
 f2c_slascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &e[l], n, info, (ftnlen)1);
 }
 i__1 = lend - 1;
 for (i__ = l;
 i__ <= i__1;
 ++i__) {
 /* Computing 2nd power */
 r__1 = e[i__];
 e[i__] = r__1 * r__1;
 /* L40: */
 }
 /* Choose between QL and QR iteration */
 if ((r__1 = d__[lend], bla_r_abs(r__1)) < (r__2 = d__[l], bla_r_abs(r__2))) {
 lend = lsv;
 l = lendsv;
 }
 if (lend >= l) {
 /* QL Iteration */
 /* Look for small subdiagonal element. */
 L50: if (l != lend) {
 i__1 = lend - 1;
 for (m = l;
 m <= i__1;
 ++m) {
 if ((r__2 = e[m], bla_r_abs(r__2)) <= eps2 * (r__1 = d__[m] * d__[m + 1], bla_r_abs(r__1))) {
 goto L70;
 }
 /* L60: */
 }
 }
 m = lend;
 L70: if (m < lend) {
 e[m] = 0.f;
 }
 p = d__[l];
 if (m == l) {
 goto L90;
 }
 /* If remaining matrix is 2 by 2, use SLAE2 to compute its */
 /* eigenvalues. */
 if (m == l + 1) {
 rte = sqrt(e[l]);
 f2c_slae2_(&d__[l], &rte, &d__[l + 1], &rt1, &rt2);
 d__[l] = rt1;
 d__[l + 1] = rt2;
 e[l] = 0.f;
 l += 2;
 if (l <= lend) {
 goto L50;
 }
 goto L150;
 }
 if (jtot == nmaxit) {
 goto L150;
 }
 ++jtot;
 /* Form shift. */
 rte = sqrt(e[l]);
 sigma = (d__[l + 1] - p) / (rte * 2.f);
 r__ = f2c_slapy2_(&sigma, &c_b32);
 sigma = p - rte / (sigma + bla_r_sign(&r__, &sigma));
 c__ = 1.f;
 s = 0.f;
 gamma = d__[m] - sigma;
 p = gamma * gamma;
 /* Inner loop */
 i__1 = l;
 for (i__ = m - 1;
 i__ >= i__1;
 --i__) {
 bb = e[i__];
 r__ = p + bb;
 if (i__ != m - 1) {
 e[i__ + 1] = s * r__;
 }
 oldc = c__;
 c__ = p / r__;
 s = bb / r__;
 oldgam = gamma;
 alpha = d__[i__];
 gamma = c__ * (alpha - sigma) - s * oldgam;
 d__[i__ + 1] = oldgam + (alpha - gamma);
 if (c__ != 0.f) {
 p = gamma * gamma / c__;
 }
 else {
 p = oldc * bb;
 }
 /* L80: */
 }
 e[l] = s * p;
 d__[l] = sigma + gamma;
 goto L50;
 /* Eigenvalue found. */
 L90: d__[l] = p;
 ++l;
 if (l <= lend) {
 goto L50;
 }
 goto L150;
 }
 else {
 /* QR Iteration */
 /* Look for small superdiagonal element. */
 L100: i__1 = lend + 1;
 for (m = l;
 m >= i__1;
 --m) {
 if ((r__2 = e[m - 1], bla_r_abs(r__2)) <= eps2 * (r__1 = d__[m] * d__[m - 1], bla_r_abs(r__1))) {
 goto L120;
 }
 /* L110: */
 }
 m = lend;
 L120: if (m > lend) {
 e[m - 1] = 0.f;
 }
 p = d__[l];
 if (m == l) {
 goto L140;
 }
 /* If remaining matrix is 2 by 2, use SLAE2 to compute its */
 /* eigenvalues. */
 if (m == l - 1) {
 rte = sqrt(e[l - 1]);
 f2c_slae2_(&d__[l], &rte, &d__[l - 1], &rt1, &rt2);
 d__[l] = rt1;
 d__[l - 1] = rt2;
 e[l - 1] = 0.f;
 l += -2;
 if (l >= lend) {
 goto L100;
 }
 goto L150;
 }
 if (jtot == nmaxit) {
 goto L150;
 }
 ++jtot;
 /* Form shift. */
 rte = sqrt(e[l - 1]);
 sigma = (d__[l - 1] - p) / (rte * 2.f);
 r__ = f2c_slapy2_(&sigma, &c_b32);
 sigma = p - rte / (sigma + bla_r_sign(&r__, &sigma));
 c__ = 1.f;
 s = 0.f;
 gamma = d__[m] - sigma;
 p = gamma * gamma;
 /* Inner loop */
 i__1 = l - 1;
 for (i__ = m;
 i__ <= i__1;
 ++i__) {
 bb = e[i__];
 r__ = p + bb;
 if (i__ != m) {
 e[i__ - 1] = s * r__;
 }
 oldc = c__;
 c__ = p / r__;
 s = bb / r__;
 oldgam = gamma;
 alpha = d__[i__ + 1];
 gamma = c__ * (alpha - sigma) - s * oldgam;
 d__[i__] = oldgam + (alpha - gamma);
 if (c__ != 0.f) {
 p = gamma * gamma / c__;
 }
 else {
 p = oldc * bb;
 }
 /* L130: */
 }
 e[l - 1] = s * p;
 d__[l] = sigma + gamma;
 goto L100;
 /* Eigenvalue found. */
 L140: d__[l] = p;
 --l;
 if (l >= lend) {
 goto L100;
 }
 goto L150;
 }
 /* Undo scaling if necessary */
 L150: if (iscale == 1) {
 i__1 = lendsv - lsv + 1;
 f2c_slascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &d__[lsv], n, info, (ftnlen)1);
 }
 if (iscale == 2) {
 i__1 = lendsv - lsv + 1;
 f2c_slascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &d__[lsv], n, info, (ftnlen)1);
 }
 /* Check for no convergence to an eigenvalue after a total */
 /* of N*MAXIT iterations. */
 if (jtot < nmaxit) {
 goto L10;
 }
 i__1 = *n - 1;
 for (i__ = 1;
 i__ <= i__1;
 ++i__) {
 if (e[i__] != 0.f) {
 ++(*info);
 }
 /* L160: */
 }
 goto L180;
 /* Sort eigenvalues in increasing order. */
 L170: f2c_slasrt_("I", n, &d__[1], info, (ftnlen)1);
 L180: return 0;
 /* End of SSTERF */
 }
 /* f2c_ssterf_ */
 
