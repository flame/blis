/* f2c_zladiv.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b ZLADIV performs bla_scomplex division in bla_real arithmetic, avoiding unnecessary overflow. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download ZLADIV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zladiv. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zladiv. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zladiv. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* COMPLEX*16 FUNCTION ZLADIV( X, Y ) */
 /* .. Scalar Arguments .. */
 /* COMPLEX*16 X, Y */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > ZLADIV := X / Y, where X and Y are bla_scomplex. The computation of X / Y */
 /* > will not overflow on an intermediary step unless the results */
 /* > overflows. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] X */
 /* > \verbatim */
 /* > X is COMPLEX*16 */
 /* > \endverbatim */
 /* > */
 /* > \param[in] Y */
 /* > \verbatim */
 /* > Y is COMPLEX*16 */
 /* > The bla_scomplex scalars X and Y. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplex16OTHERauxiliary */
 /* ===================================================================== */
 void f2c_zladiv_(bla_dcomplex * ret_val, bla_dcomplex *x, bla_dcomplex *y) {
 /* System generated locals */
 bla_double d__1, d__2, d__3, d__4;
 bla_dcomplex z__1;
 /* Builtin functions */
 /* Local variables */
 bla_double zi, zr;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Local Scalars .. */
 /* .. */
 /* .. External Subroutines .. */
 /* .. */
 /* .. Intrinsic Functions .. */
 /* .. */
 /* .. Executable Statements .. */
 d__1 = x->real;
 d__2 = bla_d_imag(x);
 d__3 = y->real;
 d__4 = bla_d_imag(y);
 f2c_dladiv_(&d__1, &d__2, &d__3, &d__4, &zr, &zi);
 z__1.real = zr, z__1.imag = zi;
 ret_val->real = z__1.real, ret_val->imag = z__1.imag;
 return ;
 /* End of ZLADIV */
 }
 /* f2c_zladiv_ */
 
