/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

extern
complex cdotc_(integer *n, complex *cx, integer *incx, complex *cy, integer *incy);

VOID cdotc_f2c_(complex *r, integer *n, complex *cx, integer *incx, complex *cy, integer *incy)
{
  *r = cdotc_(n, cx, incx, cy, incy);
}

extern
doublecomplex zdotc_(integer *n, doublecomplex *zx, integer *incx, doublecomplex *zy, integer *incy);
VOID zdotc_f2c_(doublecomplex *r, integer *n, doublecomplex *cx, integer *incx, doublecomplex *cy, integer *incy)
{
  *r = zdotc_(n, cx, incx, cy, incy);
}





void PASTEF77(c,dotu_f2c)
     (
             bla_scomplex* r,
       const bla_integer*  n,
             bla_scomplex* x, const bla_integer* incx,
             bla_scomplex* y, const bla_integer* incy
     )
{
	*r = PASTEF77(c,dotu)( n, x, incx, y, incy );
}

void PASTEF77(z,dotu_f2c)
     (
             bla_dcomplex* r,
       const bla_integer*  n,
             bla_dcomplex* x, const bla_integer* incx,
             bla_dcomplex* y, const bla_integer* incy
     )
{
	*r = PASTEF77(z,dotu)( n, x, incx, y, incy );
}


#ifdef BLIS_ENABLE_BLAS

scomplex cdotc_( integer *n, complex *cx, integer *incx, complex *cy, integer *incy);

VOID cdotc_f2c_(complex *r, integer *n, complex *cx, integer *incx, complex *cy, integer *incy)
{
	*r = cdotc_(n, cx, incx, cy, incy);
}




double bla_cdotc_f2c_(const bla_scomplex *z)
{
	return( bla_f__cabs( bli_creal( *z ),
	                     bli_cimag( *z ) ) );
}

#endif

