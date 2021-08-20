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
#include "f77_dot_sub.h"

#ifdef BLIS_ENABLE_CBLAS

//
// Define CBLAS subrotine wrapper interfaces.
//
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
void PASTEF773(ch,blasname,chc,sub) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy, \
             ftype*   rval  \
     ) \
{ \
	*rval = PASTEF772(ch,blasname,chc) \
	( \
	  n, \
	  x, incx, \
	  y, incy \
	); \
}

INSERT_GENTFUNCDOTR_BLAS( dot, NULL )

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL

INSERT_GENTFUNCDOTC_BLAS( dot, NULL )

#else

//
// Define CBLAS subrotine wrapper interfaces for complex types.
// For the "intel" complex return type, pass a hidden first parameter
// (by address).
//
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
void PASTEF773(ch,blasname,chc,sub) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy, \
             ftype*   rval  \
     ) \
{ \
	PASTEF772(ch,blasname,chc) \
	( \
	  rval, \
	  n, \
	  x, incx, \
	  y, incy \
	); \
}

INSERT_GENTFUNCDOTC_BLAS( dot, NULL )

#endif

// -- "Black sheep" dot product function definitions --

// Input vectors stored in single precision, computed in double precision,
// with result returned in single precision.
void PASTEF772(sds,dot,sub)
     (
       const f77_int* n,
       const float*  sb,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy,
             float*   rval
     )
{
	*rval = PASTEF77(sds,dot)
	(
	  n,
	  sb,
	  x, incx,
	  y, incy
	);
}

// Input vectors stored in single precision, computed in double precision,
// with result returned in double precision.
void PASTEF772(ds,dot,sub)
     (
       const f77_int* n,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy,
             double*  rval
     )
{
	*rval = PASTEF77(ds,dot)
	(
	  n,
	  x, incx,
	  y, incy
	);
}

#endif

