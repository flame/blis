/* f2c_ilaenv.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* Table of constant values */
 static bla_integer c__1 = 1;
 static bla_real c_b174 = 0.f;
 static bla_real c_b175 = 1.f;
 static bla_integer c__0 = 0;
 /* > \brief \b ILAENV */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download ILAENV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ilaenv. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ilaenv. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ilaenv. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* INTEGER FUNCTION ILAENV( ISPEC, NAME, OPTS, N1, N2, N3, N4 ) */
 /* .. Scalar Arguments .. */
 /* CHARACTER*( * ) NAME, OPTS */
 /* INTEGER ISPEC, N1, N2, N3, N4 */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > ILAENV is called from the LAPACK routines to choose problem-dependent */
 /* > parameters for the local environment. See ISPEC for a description of */
 /* > the parameters. */
 /* > */
 /* > ILAENV returns an INTEGER */
 /* > if ILAENV >= 0: ILAENV returns the value of the parameter specified by ISPEC */
 /* > if ILAENV < 0: if ILAENV = -k, the k-th argument had an illegal value. */
 /* > */
 /* > This version provides a set of parameters which should give good, */
 /* > but not optimal, performance on many of the currently available */
 /* > computers. Users are encouraged to modify this subroutine to set */
 /* > the tuning parameters for their particular machine using the option */
 /* > and problem size information in the arguments. */
 /* > */
 /* > This routine will not function correctly if it is converted to all */
 /* > lower case. Converting it to all upper case is allowed. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] ISPEC */
 /* > \verbatim */
 /* > ISPEC is INTEGER */
 /* > Specifies the parameter to be returned as the value of */
 /* > ILAENV. */
 /* > = 1: the optimal blocksize;
 if this value is 1, an unblocked */
 /* > algorithm will give the best performance. */
 /* > = 2: the minimum block size for which the block routine */
 /* > should be used;
 if the usable block size is less than */
 /* > this value, an unblocked routine should be used. */
 /* > = 3: the crossover point (in a block routine, for N less */
 /* > than this value, an unblocked routine should be used) */
 /* > = 4: the number of shifts, used in the nonsymmetric */
 /* > eigenvalue routines (DEPRECATED) */
 /* > = 5: the minimum column dimension for blocking to be used;
 */
 /* > rectangular blocks must have dimension at least k by m, */
 /* > where k is given by ILAENV(2,...) and m by ILAENV(5,...) */
 /* > = 6: the crossover point for the SVD (when reducing an m by n */
 /* > matrix to bidiagonal form, if bla_a_max(m,n)/bla_a_min(m,n) exceeds */
 /* > this value, a QR factorization is used first to reduce */
 /* > the matrix to a triangular form.) */
 /* > = 7: the number of processors */
 /* > = 8: the crossover point for the multishift QR method */
 /* > for nonsymmetric eigenvalue problems (DEPRECATED) */
 /* > = 9: maximum size of the subproblems at the bottom of the */
 /* > computation tree in the divide-and-conquer algorithm */
 /* > (used by xGELSD and xGESDD) */
 /* > =10: ieee infinity and NaN arithmetic can be trusted not to trap */
 /* > =11: infinity arithmetic can be trusted not to trap */
 /* > 12 <= ISPEC <= 16: */
 /* > xHSEQR or related subroutines, */
 /* > see IPARMQ for detailed explanation */
 /* > \endverbatim */
 /* > */
 /* > \param[in] NAME */
 /* > \verbatim */
 /* > NAME is CHARACTER*(*) */
 /* > The name of the calling subroutine, in either upper case or */
 /* > lower case. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] OPTS */
 /* > \verbatim */
 /* > OPTS is CHARACTER*(*) */
 /* > The character options to the subroutine NAME, concatenated */
 /* > into a single character string. For example, UPLO = 'U', */
 /* > TRANS = 'T', and DIAG = 'N' for a triangular routine would */
 /* > be specified as OPTS = 'UTN'. */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N1 */
 /* > \verbatim */
 /* > N1 is INTEGER */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N2 */
 /* > \verbatim */
 /* > N2 is INTEGER */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N3 */
 /* > \verbatim */
 /* > N3 is INTEGER */
 /* > \endverbatim */
 /* > */
 /* > \param[in] N4 */
 /* > \verbatim */
 /* > N4 is INTEGER */
 /* > Problem dimensions for the subroutine NAME;
 these may not all */
 /* > be required. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* > \par Further Details: */
 /* ===================== */
 /* > */
 /* > \verbatim */
 /* > */
 /* > The following conventions have been used when calling ILAENV from the */
 /* > LAPACK routines: */
 /* > 1) OPTS is a concatenation of all of the character options to */
 /* > subroutine NAME, in the same order that they appear in the */
 /* > argument list for NAME, even if they are not used in determining */
 /* > the value of the parameter specified by ISPEC. */
 /* > 2) The problem dimensions N1, N2, N3, N4 are specified in the order */
 /* > that they appear in the argument list for NAME. N1 is used */
 /* > first, N2 second, and so on, and unused problem dimensions are */
 /* > passed a value of -1. */
 /* > 3) The parameter value returned by ILAENV is checked for validity in */
 /* > the calling subroutine. For example, ILAENV is used to retrieve */
 /* > the optimal blocksize for STRTRI as follows: */
 /* > */
 /* > NB = ILAENV( 1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 ) */
 /* > IF( NB.LE.1 ) NB = MAX( 1, N ) */
 /* > \endverbatim */
 /* > */
 /* ===================================================================== */
 bla_integer f2c_ilaenv_(bla_integer *ispec, char *name__, char *opts, bla_integer *n1, bla_integer *n2, bla_integer *n3, bla_integer *n4, ftnlen name_len, ftnlen opts_len) {
 /* System generated locals */
 bla_integer ret_val;
 /* Builtin functions */
 /* Local variables */
 bla_logical twostage;
 bla_integer i__;
 char c1[1], c2[2], c3[3], c4[2];
 bla_integer ic, nb, iz, nx;
 bla_logical cname;
 bla_integer nbmin;
 bla_logical sname;
 char subnam[16];
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Local Scalars .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. External Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 switch (*ispec) {
 case 1: goto L10;
 case 2: goto L10;
 case 3: goto L10;
 case 4: goto L80;
 case 5: goto L90;
 case 6: goto L100;
 case 7: goto L110;
 case 8: goto L120;
 case 9: goto L130;
 case 10: goto L140;
 case 11: goto L150;
 case 12: goto L160;
 case 13: goto L160;
 case 14: goto L160;
 case 15: goto L160;
 case 16: goto L160;
 }
 /* Invalid value for ISPEC */
 ret_val = -1;
 return ret_val;
 L10: /* Convert NAME to upper case if the first character is lower case. */
 ret_val = 1;
 bla_s_copy(subnam, name__, (ftnlen)16, name_len);
 ic = *(unsigned char *)subnam;
 iz = 'Z';
 if (iz == 90 || iz == 122) {
 /* ASCII character set */
 if (ic >= 97 && ic <= 122) {
 *(unsigned char *)subnam = (char) (ic - 32);
 for (i__ = 2;
 i__ <= 6;
 ++i__) {
 ic = *(unsigned char *)&subnam[i__ - 1];
 if (ic >= 97 && ic <= 122) {
 *(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);
 }
 /* L20: */
 }
 }
 }
 else if (iz == 233 || iz == 169) {
 /* EBCDIC character set */
 if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 162 && ic <= 169) {
 *(unsigned char *)subnam = (char) (ic + 64);
 for (i__ = 2;
 i__ <= 6;
 ++i__) {
 ic = *(unsigned char *)&subnam[i__ - 1];
 if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 162 && ic <= 169) {
 *(unsigned char *)&subnam[i__ - 1] = (char) (ic + 64);
 }
 /* L30: */
 }
 }
 }
 else if (iz == 218 || iz == 250) {
 /* Prime machines: ASCII+128 */
 if (ic >= 225 && ic <= 250) {
 *(unsigned char *)subnam = (char) (ic - 32);
 for (i__ = 2;
 i__ <= 6;
 ++i__) {
 ic = *(unsigned char *)&subnam[i__ - 1];
 if (ic >= 225 && ic <= 250) {
 *(unsigned char *)&subnam[i__ - 1] = (char) (ic - 32);
 }
 /* L40: */
 }
 }
 }
 *(unsigned char *)c1 = *(unsigned char *)subnam;
 sname = *(unsigned char *)c1 == 'S' || *(unsigned char *)c1 == 'D';
 cname = *(unsigned char *)c1 == 'C' || *(unsigned char *)c1 == 'Z';
 if (! (cname || sname)) {
 return ret_val;
 }
 bla_s_copy(c2, subnam + 1, (ftnlen)2, (ftnlen)2);
 bla_s_copy(c3, subnam + 3, (ftnlen)3, (ftnlen)3);
 bla_s_copy(c4, c3 + 1, (ftnlen)2, (ftnlen)2);
 twostage = bla_i_len(subnam, (ftnlen)16) >= 11 && *(unsigned char *)&subnam[ 10] == '2';
 switch (*ispec) {
 case 1: goto L50;
 case 2: goto L60;
 case 3: goto L70;
 }
 L50: /* ISPEC = 1: block size */
 /* In these examples, separate code is provided for setting NB for */
 /* bla_real and bla_scomplex. We assume that NB will take the same value in */
 /* single or double precision. */
 nb = 1;
 if (bla_s_cmp(subnam + 1, "LAORH", (ftnlen)5, (ftnlen)5) == 0) {
 /* This is for *LAORHR_GETRFNP routine */
 if (sname) {
 nb = 32;
 }
 else {
 nb = 32;
 }
 }
 else if (bla_s_cmp(c2, "GE", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 64;
 }
 else {
 nb = 64;
 }
 }
 else if (bla_s_cmp(c3, "QRF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "RQF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "LQF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "QLF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 32;
 }
 else {
 nb = 32;
 }
 }
 else if (bla_s_cmp(c3, "QR ", (ftnlen)3, (ftnlen)3) == 0) {
 if (*n3 == 1) {
 if (sname) {
 /* M*N */
 if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
 nb = *n1;
 }
 else {
 nb = 32768 / *n2;
 }
 }
 else {
 if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
 nb = *n1;
 }
 else {
 nb = 32768 / *n2;
 }
 }
 }
 else {
 if (sname) {
 nb = 1;
 }
 else {
 nb = 1;
 }
 }
 }
 else if (bla_s_cmp(c3, "LQ ", (ftnlen)3, (ftnlen)3) == 0) {
 if (*n3 == 2) {
 if (sname) {
 /* M*N */
 if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
 nb = *n1;
 }
 else {
 nb = 32768 / *n2;
 }
 }
 else {
 if (*n1 * *n2 <= 131072 || *n1 <= 8192) {
 nb = *n1;
 }
 else {
 nb = 32768 / *n2;
 }
 }
 }
 else {
 if (sname) {
 nb = 1;
 }
 else {
 nb = 1;
 }
 }
 }
 else if (bla_s_cmp(c3, "HRD", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 32;
 }
 else {
 nb = 32;
 }
 }
 else if (bla_s_cmp(c3, "BRD", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 32;
 }
 else {
 nb = 32;
 }
 }
 else if (bla_s_cmp(c3, "TRI", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 64;
 }
 else {
 nb = 64;
 }
 }
 }
 else if (bla_s_cmp(c2, "PO", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 64;
 }
 else {
 nb = 64;
 }
 }
 }
 else if (bla_s_cmp(c2, "SY", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 if (twostage) {
 nb = 192;
 }
 else {
 nb = 64;
 }
 }
 else {
 if (twostage) {
 nb = 192;
 }
 else {
 nb = 64;
 }
 }
 }
 else if (sname && bla_s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
 nb = 32;
 }
 else if (sname && bla_s_cmp(c3, "GST", (ftnlen)3, (ftnlen)3) == 0) {
 nb = 64;
 }
 }
 else if (cname && bla_s_cmp(c2, "HE", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (twostage) {
 nb = 192;
 }
 else {
 nb = 64;
 }
 }
 else if (bla_s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
 nb = 32;
 }
 else if (bla_s_cmp(c3, "GST", (ftnlen)3, (ftnlen)3) == 0) {
 nb = 64;
 }
 }
 else if (sname && bla_s_cmp(c2, "OR", (ftnlen)2, (ftnlen)2) == 0) {
 if (*(unsigned char *)c3 == 'G') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nb = 32;
 }
 }
 else if (*(unsigned char *)c3 == 'M') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nb = 32;
 }
 }
 }
 else if (cname && bla_s_cmp(c2, "UN", (ftnlen)2, (ftnlen)2) == 0) {
 if (*(unsigned char *)c3 == 'G') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nb = 32;
 }
 }
 else if (*(unsigned char *)c3 == 'M') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nb = 32;
 }
 }
 }
 else if (bla_s_cmp(c2, "GB", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 if (*n4 <= 64) {
 nb = 1;
 }
 else {
 nb = 32;
 }
 }
 else {
 if (*n4 <= 64) {
 nb = 1;
 }
 else {
 nb = 32;
 }
 }
 }
 }
 else if (bla_s_cmp(c2, "PB", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 if (*n2 <= 64) {
 nb = 1;
 }
 else {
 nb = 32;
 }
 }
 else {
 if (*n2 <= 64) {
 nb = 1;
 }
 else {
 nb = 32;
 }
 }
 }
 }
 else if (bla_s_cmp(c2, "TR", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRI", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 64;
 }
 else {
 nb = 64;
 }
 }
 else if (bla_s_cmp(c3, "EVC", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 64;
 }
 else {
 nb = 64;
 }
 }
 }
 else if (bla_s_cmp(c2, "LA", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "UUM", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 64;
 }
 else {
 nb = 64;
 }
 }
 }
 else if (sname && bla_s_cmp(c2, "ST", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "EBZ", (ftnlen)3, (ftnlen)3) == 0) {
 nb = 1;
 }
 }
 else if (bla_s_cmp(c2, "GG", (ftnlen)2, (ftnlen)2) == 0) {
 nb = 32;
 if (bla_s_cmp(c3, "HD3", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nb = 32;
 }
 else {
 nb = 32;
 }
 }
 }
 ret_val = nb;
 return ret_val;
 L60: /* ISPEC = 2: minimum block size */
 nbmin = 2;
 if (bla_s_cmp(c2, "GE", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "QRF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "RQF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "LQF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "QLF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nbmin = 2;
 }
 else {
 nbmin = 2;
 }
 }
 else if (bla_s_cmp(c3, "HRD", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nbmin = 2;
 }
 else {
 nbmin = 2;
 }
 }
 else if (bla_s_cmp(c3, "BRD", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nbmin = 2;
 }
 else {
 nbmin = 2;
 }
 }
 else if (bla_s_cmp(c3, "TRI", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nbmin = 2;
 }
 else {
 nbmin = 2;
 }
 }
 }
 else if (bla_s_cmp(c2, "SY", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nbmin = 8;
 }
 else {
 nbmin = 8;
 }
 }
 else if (sname && bla_s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
 nbmin = 2;
 }
 }
 else if (cname && bla_s_cmp(c2, "HE", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
 nbmin = 2;
 }
 }
 else if (sname && bla_s_cmp(c2, "OR", (ftnlen)2, (ftnlen)2) == 0) {
 if (*(unsigned char *)c3 == 'G') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nbmin = 2;
 }
 }
 else if (*(unsigned char *)c3 == 'M') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nbmin = 2;
 }
 }
 }
 else if (cname && bla_s_cmp(c2, "UN", (ftnlen)2, (ftnlen)2) == 0) {
 if (*(unsigned char *)c3 == 'G') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nbmin = 2;
 }
 }
 else if (*(unsigned char *)c3 == 'M') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nbmin = 2;
 }
 }
 }
 else if (bla_s_cmp(c2, "GG", (ftnlen)2, (ftnlen)2) == 0) {
 nbmin = 2;
 if (bla_s_cmp(c3, "HD3", (ftnlen)3, (ftnlen)3) == 0) {
 nbmin = 2;
 }
 }
 ret_val = nbmin;
 return ret_val;
 L70: /* ISPEC = 3: crossover point */
 nx = 0;
 if (bla_s_cmp(c2, "GE", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "QRF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "RQF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "LQF", (ftnlen)3, (ftnlen)3) == 0 || bla_s_cmp(c3, "QLF", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nx = 128;
 }
 else {
 nx = 128;
 }
 }
 else if (bla_s_cmp(c3, "HRD", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nx = 128;
 }
 else {
 nx = 128;
 }
 }
 else if (bla_s_cmp(c3, "BRD", (ftnlen)3, (ftnlen)3) == 0) {
 if (sname) {
 nx = 128;
 }
 else {
 nx = 128;
 }
 }
 }
 else if (bla_s_cmp(c2, "SY", (ftnlen)2, (ftnlen)2) == 0) {
 if (sname && bla_s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
 nx = 32;
 }
 }
 else if (cname && bla_s_cmp(c2, "HE", (ftnlen)2, (ftnlen)2) == 0) {
 if (bla_s_cmp(c3, "TRD", (ftnlen)3, (ftnlen)3) == 0) {
 nx = 32;
 }
 }
 else if (sname && bla_s_cmp(c2, "OR", (ftnlen)2, (ftnlen)2) == 0) {
 if (*(unsigned char *)c3 == 'G') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nx = 128;
 }
 }
 }
 else if (cname && bla_s_cmp(c2, "UN", (ftnlen)2, (ftnlen)2) == 0) {
 if (*(unsigned char *)c3 == 'G') {
 if (bla_s_cmp(c4, "QR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "RQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "LQ", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "QL", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "HR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp( c4, "TR", (ftnlen)2, (ftnlen)2) == 0 || bla_s_cmp(c4, "BR", (ftnlen)2, (ftnlen)2) == 0) {
 nx = 128;
 }
 }
 }
 else if (bla_s_cmp(c2, "GG", (ftnlen)2, (ftnlen)2) == 0) {
 nx = 128;
 if (bla_s_cmp(c3, "HD3", (ftnlen)3, (ftnlen)3) == 0) {
 nx = 128;
 }
 }
 ret_val = nx;
 return ret_val;
 L80: /* ISPEC = 4: number of shifts (used by xHSEQR) */
 ret_val = 6;
 return ret_val;
 L90: /* ISPEC = 5: minimum column dimension (not used) */
 ret_val = 2;
 return ret_val;
 L100: /* ISPEC = 6: crossover point for SVD (used by xGELSS and xGESVD) */
 ret_val = (bla_integer) ((bla_real) bla_a_min(*n1,*n2) * 1.6f);
 return ret_val;
 L110: /* ISPEC = 7: number of processors (not used) */
 ret_val = 1;
 return ret_val;
 L120: /* ISPEC = 8: crossover point for multishift (used by xHSEQR) */
 ret_val = 50;
 return ret_val;
 L130: /* ISPEC = 9: maximum size of the subproblems at the bottom of the */
 /* computation tree in the divide-and-conquer algorithm */
 /* (used by xGELSD and xGESDD) */
 ret_val = 25;
 return ret_val;
 L140: /* ISPEC = 10: ieee and infinity NaN arithmetic can be trusted not to trap */
 /* ILAENV = 0 */
 ret_val = 1;
 if (ret_val == 1) {
 ret_val = f2c_ieeeck_(&c__1, &c_b174, &c_b175);
 }
 return ret_val;
 L150: /* ISPEC = 11: ieee infinity arithmetic can be trusted not to trap */
 /* ILAENV = 0 */
 ret_val = 1;
 if (ret_val == 1) {
 ret_val = f2c_ieeeck_(&c__0, &c_b174, &c_b175);
 }
 return ret_val;
 L160: /* 12 <= ISPEC <= 16: xHSEQR or related subroutines. */
 ret_val = f2c_iparmq_(ispec, name__, opts, n1, n2, n3, n4, name_len, opts_len) ;
 return ret_val;
 /* End of ILAENV */
 }
 /* f2c_ilaenv_ */
 
