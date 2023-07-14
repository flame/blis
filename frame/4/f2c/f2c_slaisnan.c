/* f2c_slaisnan.f -- translated by f2c (version 20100827). You must link the resulting object file with libf2c: on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm or, if you install libf2c.a in a standard place, with -lf2c -lm -- in that order, at the end of the command line, as in cc *.o -lf2c -lm Source for libf2c is in /netlib/f2c/libf2c.zip, e.g., http://www.netlib.org/f2c/libf2c.zip */
 #include "blis.h" /* > \brief \b SLAISNAN tests input for NaN by comparing two arguments for inequality. */
 /* =========== DOCUMENTATION =========== */
 /* Online html documentation available at */
 /* http://www.netlib.org/lapack/explore-html/ */
 /* > \htmlonly */
 /* > Download SLAISNAN + dependencies */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slaisna n.f"> */
 /* > [TGZ]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slaisna n.f"> */
 /* > [ZIP]</a> */
 /* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slaisna n.f"> */
 /* > [TXT]</a> */
 /* > \endhtmlonly */
 /* Definition: */
 /* =========== */
 /* LOGICAL FUNCTION SLAISNAN( SIN1, SIN2 ) */
 /* .. Scalar Arguments .. */
 /* REAL SIN1, SIN2 */
 /* .. */
 /* > \par Purpose: */
 /* ============= */
 /* > */
 /* > \verbatim */
 /* > */
 /* > This routine is not for general use. It exists solely to avoid */
 /* > over-optimization in SISNAN. */
 /* > */
 /* > SLAISNAN checks for NaNs by comparing its two arguments for */
 /* > inequality. NaN is the only floating-point value where NaN != NaN */
 /* > returns .TRUE. To check for NaNs, pass the same variable as both */
 /* > arguments. */
 /* > */
 /* > A compiler must assume that the two arguments are */
 /* > not the same variable, and the test will not be optimized away. */
 /* > Interprocedural or whole-program optimization may delete this */
 /* > test. The ISNAN functions will be replaced by the correct */
 /* > Fortran 03 intrinsic once the intrinsic is widely available. */
 /* > \endverbatim */
 /* Arguments: */
 /* ========== */
 /* > \param[in] SIN1 */
 /* > \verbatim */
 /* > SIN1 is REAL */
 /* > \endverbatim */
 /* > */
 /* > \param[in] SIN2 */
 /* > \verbatim */
 /* > SIN2 is REAL */
 /* > Two numbers to compare for inequality. */
 /* > \endverbatim */
 /* Authors: */
 /* ======== */
 /* > \author Univ. of Tennessee */
 /* > \author Univ. of California Berkeley */
 /* > \author Univ. of Colorado Denver */
 /* > \author NAG Ltd. */
 /* > \ingroup OTHERauxiliary */
 /* ===================================================================== */
 bla_logical f2c_slaisnan_(bla_real *sin1, bla_real *sin2) {
 /* System generated locals */
 bla_logical ret_val;
 /* -- LAPACK auxiliary routine -- */
 /* -- LAPACK is a software package provided by Univ. of Tennessee, -- */
 /* -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
 /* .. Scalar Arguments .. */
 /* .. */
 /* ===================================================================== */
 /* .. Executable Statements .. */
 ret_val = *sin1 != *sin2;
 return ret_val;
 }
 /* f2c_slaisnan_ */
 
