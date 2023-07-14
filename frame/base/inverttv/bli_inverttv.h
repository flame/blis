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


//
// Prototype object-based check functions.
//

void bli_inverttv_check
     (
             double thresh,
       const obj_t* x
     );

// -----------------------------------------------------------------------------

//
// Prototype typed API function types.
//

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH2(ch,opname,tsuf)) \
     ( \
             double  thresh, \
             dim_t   m, \
             ctype*  x, inc_t incx  \
     );

INSERT_GENTDEF( inverttv )

// -----------------------------------------------------------------------------

//
// Prototype function pointer query interface.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
PASTECH(opname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt );

GENPROT( inverttv )

// -----------------------------------------------------------------------------

//
// Prototype object-based interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC0(opname) \
     ( \
             double  thresh, \
       const obj_t*  x  \
     );

GENPROT( inverttv )

// -----------------------------------------------------------------------------

//
// Prototype BLAS-like interfaces with typed operands.
//

#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
             double  thresh, \
             dim_t   m, \
             ctype*  x, inc_t incx  \
     );

INSERT_GENTPROTR_BASIC0( inverttv )

