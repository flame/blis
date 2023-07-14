/* f2c_cladiv.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b CLADIV performs bla_scomplex division in bla_real arithmetic, avoiding unnecessary overflow. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download CLADIV + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cladiv. f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cladiv. f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cladiv. f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* COMPLEX FUNCTION CLADIV( X, Y ) */
 /* .. Scalar Arguments .. */
 /* COMPLEX X, Y */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > CLADIV := X / Y, where X and Y are bla_scomplex. The computation of X / Y */
 /* > will not overflow on an intermediary step unless the results */
 /* > overflows. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] X */
 /* > \verbatim */
 /* > X is COMPLEX */
 /* > \endverbatim */
 /* > */
 /* > \param[in] Y */
 /* > \verbatim */
 /* > Y is COMPLEX */
 /* > The bla_scomplex scalars X and Y. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup bla_scomplexOTHERauxiliary */
 /* ===================================================================== */
 void f2c_cladiv_(bla_scomplex * ret_val, bla_scomplex *x, bla_scomplex *y) {
 /* System generated locals */
 bla_real r__1, r__2, r__3, r__4;
 bla_scomplex q__1;
 /* Builtin functions */
 /* Local variables */
 bla_real zi, zr;
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
 r__1 = x->real;
 r__2 = bla_r_imag(x);
 r__3 = y->real;
 r__4 = bla_r_imag(y);
 f2c_sladiv_(&r__1, &r__2, &r__3, &r__4, &zr, &zi);
 q__1.real = zr, q__1.imag = zi;
 ret_val->real = q__1.real, ret_val->imag = q__1.imag;
 return ;
 /* End of CLADIV */
 }
 /* f2c_cladiv_ */
 
