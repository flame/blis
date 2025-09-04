/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

/* scabs1.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ bla_real PASTEF77S(s,cabs1)(bla_scomplex *z)
{
   if ( bli_creal(*z) == 0.0f && bli_cimag(*z) == 0.0f )
   {
      /*If input is zero, return zero.
        As the else part returns -0.0 */
      return 0.0f;
   }
   else
   {
      return bli_fabs( bli_creal( *z ) ) +
           bli_fabs( bli_cimag( *z ) ); /* code */
   }
} /* scabs1_ */

/* dcabs1.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ bla_double PASTEF77S(d,cabs1)(bla_dcomplex *z)
{
   if ( bli_creal(*z) == 0.0 && bli_cimag(*z) == 0.0 )
   {
      /*If input is zero, return zero.
        As the else part returns -0.0 */
      return 0.0;
   }
   else
   {
      return bli_fabs( bli_zreal( *z ) ) +
           bli_fabs( bli_zimag( *z ) );
   }

} /* dcabs1_ */


#ifdef BLIS_ENABLE_BLAS

/* Subroutine */ bla_real PASTEF77(s,cabs1)(bla_scomplex *z)
{
  return PASTEF77S(s,cabs1)(z);
}
/* Subroutine */ bla_double PASTEF77(d,cabs1)(bla_dcomplex *z)
{
  return PASTEF77S(d,cabs1)(z);
}

#endif

