/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

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

#ifdef BLIS_ENABLE_BLAS

//
// Define functions that map:
//   f2c_cdotu_() -> cdotu_()
//   f2c_cdotc_() -> cdotc_()
//   f2c_zdotu_() -> zdotu_()
//   f2c_zdotc_() -> zdotc_()
//
// When certain code (e.g. LAPACK) is converted to C with f2c, calls to complex
// dot product "return" the dot product value in a "hidden" argument that f2c
// inserts before all others. These functions adhere to that same calling
// convention, but then immediately call the equivalent BLAS function using
// normal return value semantics. (Using these wrappers is easier than fixing
// the code emitted by f2c.)
//
// NOTE: The blis_conjx and blisname macro arguments are unused here.
//
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
void PASTEF2C2(ch,blasname,chc) \
     ( \
             ftype*   rho, \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
	/* Call BLAS interface. */ \
	*rho = PASTEF772(ch,blasname,chc) \
	( \
	  n, \
	  x, incx, \
	  y, incy  \
	); \
}

INSERT_GENTFUNCDOTC_BLAS( dot, NULL )


#endif

